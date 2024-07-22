from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import secretflow.utils.ndarray_encoding as ndarray_encoding
from secretflow.device import PYU, DeviceObject, PYUObject, proxy, reveal
from secretflow.security.aggregation import Aggregator
from secretflow.security.aggregation._utils import is_nesting_list
from sympy.ntheory import factorint

import random

class _TA:
    def __init__(self):
        '''generate p and g'''
        def find_primitive_root(p):          
            # 计算 p-1 的所有质因子
            factors = factorint(p-1)
            factor_list = list(factors.keys())
            
            # 尝试找到一个生成元
            for g in range(2, p):
                if all(pow(g, (p-1)//q, p) != 1 for q in factor_list):
                    return g
            
            print("cannot find a generator !")
            return None
        
        self._p = 10457 # big prime
        self._g = find_primitive_root(self._p) # generator
        print(f"p is {self._p}, g is {self._g}")
    
    def get_p_g(self):
        return self._p, self._g

class _Hasher:
    def __init__(self, p: int, g: int):
        self._p = p
        self._g = g
        
    def homomor_hash(self, x):
        return pow(self._g, x, self._p)


@proxy(PYUObject) # a mask is a pyu object
class _Masker:
    def __init__(self, party, participants, fxp_bits: int):
        '''party: itself
        paerticipants: all parties'''
        self._party = party # str
        self._fxp_bits = fxp_bits

        '''generate a seed for each party, but seed for itself is not used'''
        self._seed_dict = {} # {party(pyu) : seed}, generate a seed for every party
        for party_pyu in participants:
            self._seed_dict[party_pyu.party] = np.random.SeedSequence().entropy # party_pyu.party is str
            print(f"---Party: {self._party} generate seed: {self._seed_dict[party_pyu.party]}, for party: {party_pyu.party}")
            # print(f"Type of entropy: {type(self._seed_dict[party_pyu.party])}")

    def get_seed_dict(self) -> Dict:
        return self._seed_dict

    # not used yet
    def gen_rng_received(self, seed_dicts: Dict[str, Dict[str, int]]) -> None:
        '''create PRGs of receiving from the other parties
        here, np.random.default_rng is just a PRG objective,
        we also need .integers() to generate the mask'''

        assert seed_dicts, f'seed_dicts is None or empty.'
        self._rngs_received = {
            party: np.random.default_rng(
                seed_dict[self._party]
            )
            for party, seed_dict in seed_dicts.items()
            if party != self._party
        } # the dictionary, {the other party: PRG of seed received from the other party}

    def gen_seed_received(self, seed_dicts: Dict[str, Dict[str, int]]) -> None:
        '''create seeds of receiving from the other parties'''

        assert seed_dicts, f'seed_dicts is None or empty.'
        self._seeds_received = {
            party: seed_dict[self._party]
            for party, seed_dict in seed_dicts.items()
            if party != self._party
        } # the dictionary, {the other party: PRG of seed received from the other party}

    def mask(
        self,
        data: Union[
            List[Union[pd.DataFrame, pd.Series, np.ndarray]],
            Union[pd.DataFrame, pd.Series, np.ndarray],
        ],
        weight=None,
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], np.dtype]:
        '''add mask for every element in data which is a list, using the PRGs'''
        # print(f"len of data: {len(data)}, data: {data}")

        assert data is not None, 'Data shall not be None or empty.'
        is_list = isinstance(data, list)
        if not is_list:
            data = [data]
        if weight is None:
            weight = 1
        masked_data = []
        dtype = None
        for datum in data:
            # check the data type of datum
            if isinstance(datum, (pd.DataFrame, pd.Series)):
                datum = datum.values
            assert isinstance(
                datum, np.ndarray
            ), f'Accept ndarray or dataframe/series only but got {type(datum)}'
            if dtype is None:
                dtype = datum.dtype
            else:
                assert (
                    datum.dtype == dtype
                ), f'Data should have same dtypes but got {datum.dtype} {dtype}.'
            is_float = np.issubdtype(datum.dtype, np.floating)
            if not is_float:
                assert np.issubdtype(
                    datum.dtype, np.integer
                ), f'Data type are neither integer nor float.'
                if datum.dtype != np.int64:
                    datum = datum.astype(np.int64)
            
            # Do mulitple before encoding to finite field.
            # print(f"---Party: {self._party} do encode in mask(), max value: {datum.max()}")
            masked_datum: np.ndarray = (
                ndarray_encoding.encode(datum * weight, self._fxp_bits) # Encode float ndarray to uint64 finite field. Float will times 2**fxp_bits firstly
                if is_float
                else datum * weight
            )

            '''add mask'''

            # print(f"===Party {self._party}: self._seed_dict is {self._seed_dict}, self._seeds_received is {self._seeds_received}")

            # subtract the masks sent out
            for party, seed in self._seed_dict.items():
                if party == self._party:
                    continue
                rng = np.random.default_rng(seed)
                mask = rng.integers(
                    low=np.iinfo(np.int64).min,
                    high=np.iinfo(np.int64).max,
                    size=masked_datum.shape,
                ).astype(masked_datum.dtype)
                masked_datum -= mask

            # add the masks received
            for party, seed in self._seeds_received.items():
                if party == self._party:
                    continue
                rng = np.random.default_rng(seed)
                mask = rng.integers(
                    low=np.iinfo(np.int64).min,
                    high=np.iinfo(np.int64).max,
                    size=masked_datum.shape,
                ).astype(masked_datum.dtype) # create a long random integer with the same length as the masked_datum using rng
                masked_datum += mask

            masked_data.append(masked_datum)
        
        # return a list
        if is_list:
            return masked_data, dtype
        else:
            return masked_data[0], dtype


class SecureAggregator(Aggregator):
    """The secure aggregation implementation of `Masking with One-Time Pads`.

    Warnings:
        The SecureAggregator uses :py:meth:`numpy.random.PCG64`. There are many
        discussions of whether PCG is a CSPRNG
        (e.g. https://crypto.stackexchange.com/questions/77101/is-the-pcg-prng-a-csprng-or-why-not),
        we prefer a conservative strategy unless a further security analysis came
        up. Therefore we recommend users to use a standardized CSPRNG in industrial
        scenarios.
    """
    def __init__(self, device: PYU, participants: List[PYU], fxp_bits: int = 18): # 18
        assert len(set(participants)) == len( # check if there are duplicated devices
            participants
        ), 'Should not have duplicated devices.'
        self._device = device # the device for aggregation
        self._participants = set(participants)
        self._fxp_bits = fxp_bits
        self._maskers = {
            pyu: _Masker(pyu.party, self._participants, self._fxp_bits, device=pyu) for pyu in participants
        } # a dictionary, create a _Masker for every pyu of participants
        seed_dicts = reveal(
            {pyu.party: masker.get_seed_dict() for pyu, masker in self._maskers.items()}
        ) # Get plaintext data from device, get the {pyu: its seed_dict}
        for masker in self._maskers.values(): # generate mask for every participant
            masker.gen_seed_received(seed_dicts)
        
        '''init homomorphic hash for varifying'''
        """self._p, self._g = _TA().get_p_g() # invoke TA to generate p and g
        self._hasher = _Hasher(self._p, self._g)"""
    

    def _check_data(self, data: List[PYUObject]):
        assert data, f'The data should not be None or empty.'
        assert len(data) == len(
            self._maskers
        ), f'Length of the data not equals devices: {len(data)} vs {len(self._maskers)}'
        devices_of_data = set(datum.device for datum in data)
        assert (
            devices_of_data == self._participants
        ), 'Devices of the data must be corresponding with this aggregator.'
    
    '''veritfy using homomorphic hashing'''
    def _verify_homomor(self, data: List[PYUObject]):
        '''data: data list for participants
        data_sum: data after sum
        data is decoded, data_sum is encoded'''
        # print(data)
        """hash_list = []
        for i, datum in enumerate(data):
            x = np.sum(reveal(datum))
            print(x)
            hash_list.append(self._hasher.homomor_hash())"""
        
        return True

    @classmethod
    def _is_list(cls, masked_data: Union[List, Any]) -> bool:
        is_list = isinstance(masked_data[0], list)
        for masked_datum in masked_data[1:]:
            assert (
                isinstance(masked_datum, list) == is_list
            ), f'Some data are list where some others are not.'
            assert not is_list or len(masked_datum) == len(
                masked_datum[0]
            ), f'Lengths of datum in data are different.'
        return is_list
    
    def sum(self, data: List[PYUObject], axis=None):
        # print("=================sum=================")
        def _sum(*masked_data: List[np.ndarray], dtypes: List[np.dtype], fxp_bits):
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'
            is_float = np.issubdtype(dtypes[0], np.floating)

            if is_nesting_list(masked_data):
                results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                return (
                    [ndarray_encoding.decode(result, fxp_bits) for result in results]
                    if is_float
                    else results
                )
            else:
                result = np.sum(masked_data, axis=axis)
                return ndarray_encoding.decode(result, fxp_bits) if is_float else result

        self._check_data(data)
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        for i, datum in enumerate(data):
            masked_data[i], dtypes[i] = self._maskers[datum.device].mask(datum) # add masks
        masked_data = [d.to(self._device) for d in masked_data] # send masked data to the self._device (responsible to aggregate)
        dtypes = [dtype.to(self._device) for dtype in dtypes]
        return self._device(_sum)(*masked_data, dtypes=dtypes, fxp_bits=self._fxp_bits) # self._device perfoems aggregation

    def average(self, data: List[PYUObject], axis=None, weights=None):
        # print("=================avg=================")
        def _average(
            *masked_data: List[np.ndarray], dtypes: List[np.dtype], weights, fxp_bits
        ):
            '''return the aver res and the sum res'''

            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'
            is_float = np.issubdtype(dtypes[0], np.floating)
            sum_weights = np.sum(weights, axis=axis) if weights else len(masked_data)
            if is_nesting_list(masked_data):
                sum_data = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                if is_float:
                    sum_data = [
                        ndarray_encoding.decode(sum_datum, fxp_bits)
                        for sum_datum in sum_data
                    ]
                return [element / sum_weights for element in sum_data]
            else:
                if is_float:
                    return (
                        ndarray_encoding.decode(
                            np.sum(masked_data, axis=axis), fxp_bits
                        )
                        / sum_weights
                    )
                return np.sum(masked_data, axis=axis) / sum_weights

        self._check_data(data)
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        _weights = []
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            assert len(weights) == len(
                data
            ), f'Length of the weights not equals data: {len(weights)} vs {len(data)}.'
            for i, w in enumerate(weights):
                if isinstance(w, DeviceObject):
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight is not same with the corresponding data.'
                    _weights.append(w.to(self._device))
                else:
                    _weights.append(w)
            for i, (datum, weight) in enumerate(zip(data, weights)):
                masked_data[i], dtypes[i] = self._maskers[datum.device].mask(
                    datum, weight
                )
        else:
            # print("------------weights is None--------------")
            for i, datum in enumerate(data):
                masked_data[i], dtypes[i] = self._maskers[datum.device].mask(
                    datum, weights
                )
        masked_data = [d.to(self._device) for d in masked_data]
        dtypes = [dtype.to(self._device) for dtype in dtypes]

        '''verify using homomorphic hash'''
        # encode data
        """is_right = self._verify_homomor(data) # data is decoded, data_sum is encoded 
        if is_right:
            print("\u2713 : the global model is right !")
        else:
            print("\u2717 : the global model is wrong !")"""

        return self._device(_average)(
            *masked_data, dtypes=dtypes, weights=_weights, fxp_bits=self._fxp_bits
        )
    
