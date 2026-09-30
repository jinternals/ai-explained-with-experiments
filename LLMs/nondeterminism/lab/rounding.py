"""Section 03 of the post: computers round after every addition.

Standard library only. Runs on any computer with Python 3 (Mac, Linux, Windows):
    python3 lab/rounding.py
"""
import math, random, struct, sys


def f16(x):
    """Round a number to the nearest float16, the 16-bit format models use."""
    return struct.unpack("e", struct.pack("e", x))[0]


# 0. The gap between neighbouring storable numbers grows with the number's size.
#    A format with m bits of digits steps by 2**(exponent - m): bf16 m=7, float16 m=10.
def gap(x, digit_bits):
    return 2.0 ** (math.floor(math.log2(x)) - digit_bits)

print("gap to the next storable number:")
for x in (1, 21, 2048):
    print(f"  near {x:>5}: bf16 {gap(x, 7):<10g} float16 {gap(x, 10):<10g} Python float {math.ulp(x):g}")
print(f"  near  1e20: Python float {math.ulp(1e20):g}")

# 1. Above 2048, float16 only has even numbers
print("float16 values around 2048:", f16(2047), f16(2048), f16(2049), f16(2050), f16(2051))

# 2. Same five numbers, two orders
nums = [2048, 1, 1, 1, 1]
total = 0.0
for n in nums:
    total = f16(total + n)
halves = f16(f16(2048 + 1) + f16(f16(1 + 1) + 1))
print("one pass, left to right:", total)
print("two halves, then add:   ", halves)
print("true answer:            ", sum(nums))

# 3. The article's two-line example (64-bit Python floats)
print("(0.1 + 1e20) - 1e20 =", (0.1 + 1e20) - 1e20)
print("0.1 + (1e20 - 1e20) =", 0.1 + (1e20 - 1e20))

# 4. The article's shuffle example, verbatim
vals = [1e-10, 1e-5, 1e-2, 1]
vals = vals + [-v for v in vals]
loop, builtin = [], []
random.seed(42)
for _ in range(10000):
    random.shuffle(vals)
    t = 0.0
    for v in vals:          # plain left-to-right addition, same on every Python
        t += v
    loop.append(t)
    builtin.append(sum(vals))  # sum() changed how it adds floats in 3.12 and again in 3.14
print(f"shuffle, 10,000 orders, true sum 0 (Python {sys.version.split()[0]}):")
print("  plain loop:", len(set(loop)), "different results,", loop.count(0.0), "exactly 0")
print("  sum():     ", len(set(builtin)), "different results,", builtin.count(0.0), "exactly 0")
