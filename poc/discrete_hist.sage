# Demonstration of Prio3 and Poplar1 for a particular use case: The Collector
# wants to learn the errors that Clients encounter most frequently.

from sagelib.common import I2OSP, OS2IP, Unsigned, Vec, gen_rand
from sagelib.flp_generic import Histogram, FlpGeneric
from sagelib.prg import PrgAes128
from sagelib.vdaf import run_vdaf, test_vdaf
from sagelib.vdaf_prio3 import Prio3
from sagelib.vdaf_poplar1 import Poplar1Aes128

# The DiscreteHistogram circuit works just like the Histogram circuit. The only
# difference is that, instead of interpreting the buckets as "boundaries" over a
# continuous domain, each bucket corresponds to one of a discrete set of
# possible measurements. One bucket is reserved for "unknown" measurements that
# don't correspond to any of the pre-defined buckets.
#
# NOTE The size of the proof for this circuit is linear in the number of
# buckets `N`. This can easily be improved to `O(sqrt(N))`.
class DiscreteHistogram(Histogram):

    @classmethod
    def encode(cls, measurement):
        encoded = [cls.Field(0) for _ in range(len(cls.buckets)+1)]

        # The encoded measurement is a one-hot vector, where the non-zero value
        # is the index of the measurement in the bucket sequence.
        try:
            i = cls.buckets.index(measurement)
        except ValueError:
            # The last bucket is disignated for unknown measurements.
            i = len(cls.buckets)

        # Since we're counting the frequency of each value, the non-zero value
        # is simply `1`.
        #
        # NOTE This circuit can easily be exteneded to allow for richer
        # aggregation functions, such as an average of range-bounded integers
        # for each measurement.
        encoded[i] = cls.Field(1)
        return encoded

class Prio3Aes128DiscreteHistogram(Prio3):
    @classmethod
    def with_buckets(cls, buckets: Vec[Unsigned]):
        new_cls = cls.with_prg(PrgAes128)
        new_cls.Flp = FlpGeneric \
            .with_valid(DiscreteHistogram.with_buckets(buckets))
        new_cls.ID = 0xffffffff
        return new_cls


# Prio: Clients need to be able to enumerate the error types prior to
# generating their reports.
error_types = [
    'reset',
    'thing',
    'dns',
]

prio = Prio3Aes128DiscreteHistogram \
        .with_shares(2) \
        .with_buckets(error_types)

# Run VDAF evaluation for a given set of measurements. One advantage of fixing
# the error types in advance is that we can easily count the frequency of
# unknown errors reported by clients.
test_vdaf(prio,
    # Aggregation parameter
    None,

    # Measurements
    [
        'reset',             # First Client's error
        'dns',               # Second Client's error
        'reset',             # Third Client's error
        'oh no, it broke!!', # Fourth Client's error (unknown type)
    ],

    # Expected aggregate result (frequency of each error type)
    [
        2, # 'reset'
        0, # 'thing'
        1, # 'dns'
        1, # unknown error type
    ],
)


# Poplar: The Clients don't need to be able to enumerate the error types,
# but the length of the measurements (bit-strings) needs to be fixed before
# generating reports.
#
# Suppose in this applicastion that the maximum length of any error type is
# 5 bytes, i.e., 40 bits. We will encode inputs using 40+1 bits so that we
# can accammadate variable-length inputs unambiguously.
poplar = Poplar1Aes128.with_bits(41)

# Encodes an arbitrary-length string as a 2^41-bit integer, as required for
# Poplar. The input is `10*`-padded so that we caan remove the padding later on.
def encode(val):
    shift = 41 - (len(val) * 8)
    return (OS2IP(val) << shift) | (1 << (shift-1))

def decode(encoded):
    shift = 0
    while (encoded & (1<<shift)) == 0:
        shift += 1
    unpadded = encoded >> (shift+1)
    length = 5 - int(shift / 8)
    return I2OSP(unpadded, length)

# Test that the encoding is correct.
assert decode(encode(b'reset')) == b'reset'
assert decode(encode(b'dns')) == b'dns'
assert decode(encode(b'oh no')) == b'oh no'
assert decode(encode(b'')) == b''

measurements = [
    encode(b'reset'),
    encode(b'dns'),
    encode(b'reset'),
    encode(b'oh no'),
]

# Poplar in "histogram" mode: For a given set of candidate error types, count
# how many time each error type was reported. Like Prio, this requires just one
# round of aggregation. Unlike Prio, we can't count the number of unknown
# errors.
test_vdaf(poplar,
    # Aggregation parameter
    (
        # Level of the IPDF tree to evaluate.
        #
        # NOTE The aggregators MUST NOT aggregate a set of reports at a
        # given level more than once. Otherwise, privacy violations are
        # possible.
        poplar.Idpf.BITS - 1,

        # Candidate prefixes (i.e., error types).
        [
            encode(b'reset'),
            encode(b'thing'),
            encode(b'dns'),
        ],
    ),

    # Measurements
    measurements,

    # Expected aggregate result (frequency of each candidate error)
    [
        2, # 'reset'
        0, # 'thing'
        1, # 'dns'
    ],
)

# Poplar in "heavy-hitters" mode: Compute the set of errors reported at least
# some number of times (in this case at least once). This requires multiple
# rounds of aggregation. Unlike histogram mode, unknown error types can be
# discovered.
nonces = [gen_rand(16) for _ in range(len(measurements))]
candidate_prefixes = [0b0, 0b1]
for level in range(poplar.Idpf.BITS):
    agg_param = (level, candidate_prefixes)

    # TODO refactor run_vdaf() so that we can generate the reports once and
    # aggregate them multiple times. run_vdaf() generates reports (i.e.,
    # sets of input shares) for the given measurements and aggregates them
    # with the given aggregation parameter.
    agg_result = run_vdaf(poplar, agg_param, nonces, measurements)
    pretty = ['{0:0b}'.format(prefix) for prefix in candidate_prefixes]
    print(level, pretty, agg_result)

    if level != poplar.Idpf.BITS - 1:
        next_candidate_prefixes = set()
        for (prefix, count) in zip(candidate_prefixes, agg_result):
            if count > 0:
                next_candidate_prefixes.add(prefix << 1)
                next_candidate_prefixes.add((prefix << 1) + 1)
        candidate_prefixes = sorted(list(next_candidate_prefixes))

for (prefix, count) in zip(candidate_prefixes, agg_result):
    if count > 0:
        error = decode(prefix)
        print(error, count)
