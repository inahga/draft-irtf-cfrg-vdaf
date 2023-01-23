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

# Poplar: The Clients don't need to be able to enumerate the error types,
# but the length of the measurements (bit-strings) needs to be fixed before
# generating reports.
#
# Suppose in this application that the maximum length of any error type is
# 5 bytes, i.e., 40 bits. We will encode inputs using 40+1 bits so that we
# can accommodate variable-length inputs unambiguously.
NUM_BITS = 41
poplar = Poplar1Aes128.with_bits(NUM_BITS)

# Encodes an arbitrary-length string as a 2^NUM_BITS-bit integer, as required for
# Poplar. The input is `10*`-padded so that we can remove the padding later on.
def encode(val):
    shift = NUM_BITS - (len(val) * 8)
    return (OS2IP(val) << shift) | (1 << (shift-1))

def decode(encoded):
    shift = 0
    while (encoded & (1<<shift)) == 0:
        shift += 1
    unpadded = encoded >> (shift+1)
    length = 5 - int(shift / 8)
    return I2OSP(unpadded, length)

def encode_error(error_type, client_version, client_context, origin):
    return encode(bytes([error_type]) + bytes([client_version]) + bytes([client_context]) + bytes(origin, "utf-8"))

def decode_error(raw_err):
    err = decode(raw_err)
    error_type, client_version, client_context, origin = err[0], err[1], err[2], str(err[3:])
    return error_type, client_version, client_context, origin

# Test that the encoding is correct.
assert decode(encode(b'reset')) == b'reset'
assert decode(encode(b'dns')) == b'dns'
assert decode(encode(b'oh no')) == b'oh no'
assert decode(encode(b'')) == b''

# This is a one byte (uint8_t) enumerated value representing the type of error that occurred.
error_types = [
    0x00,
    0x01,
    0x02
]

# This is a one byte (uint8_t) enumerated value representing the client version.
client_version = 0x10

# This is a one byte (uint8_t) enumerated value representing the context in which the error was generated.
client_context = 0x20

measurements = [
    encode_error(error_types[0], client_version, client_context, "A"),
    encode_error(error_types[0], client_version, client_context, "B"),
    encode_error(error_types[0], client_version, client_context, "A"),
    encode_error(error_types[0], client_version, client_context, "B"),
    encode_error(error_types[2], client_version, client_context, "C"),
    # encode(b'0A'),
    # encode(b'2A'),
    # encode(b'1B'),
    # encode(b'2B'),
    # encode(b'0A'),
    # encode(b'1B'),
    # encode(b'3C'),
]

# Poplar in "heavy-hitters" mode: Compute the set of errors reported at least
# some number of times (in this case at least once). This requires multiple
# rounds of aggregation. Unlike histogram mode, unknown error types can be
# discovered.
nonces = [gen_rand(16) for _ in range(len(measurements))]
# candidate_prefixes = [0b0, 0b1]
candidate_prefixes = [error_types[0], error_types[1], error_types[2]] # one byte known prefixes
for level in range(8, poplar.Idpf.BITS):
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
            if count > 1:
                next_candidate_prefixes.add(prefix << 1)
                next_candidate_prefixes.add((prefix << 1) + 1)
        candidate_prefixes = sorted(list(next_candidate_prefixes))

for (prefix, count) in zip(candidate_prefixes, agg_result):
    if count > 1:
        # error = decode(prefix)
        error_type, client_version, client_context, origin = decode_error(prefix)        
        print("%s had %d error %d times" % (origin, error_type, count))
