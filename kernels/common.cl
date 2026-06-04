/*
	common.cl
	Bryan Little May 2026

	shared kernel functions

	Original 64 bit algorithm by
	Yves Gallot, Dec 2020
*/


typedef struct {
	uint lo;
	uint mid;
	uint hi;
	int special;
} result;

typedef struct {
	uint lo;
	uint mid;
	uint hi;
} cl_uint96_t;

typedef struct {
	cl_uint96_t lo;
	cl_uint96_t hi;
} cl_uint192_t;

typedef struct {
	cl_uint96_t r0;
	cl_uint96_t r1;
} m2p96_t;

static inline cl_uint96_t make96_3(const uint lo, const uint mid, const uint hi)
{
	cl_uint96_t r;
	r.lo = lo;
	r.mid = mid;
	r.hi = hi;
	return r;
}

static inline cl_uint96_t zero96(void) { return make96_3(0U, 0U, 0U); }
static inline cl_uint96_t one96(void)  { return make96_3(1U, 0U, 0U); }

static inline uint is_zero96(const cl_uint96_t a)
{
	return (a.lo == 0U) && (a.mid == 0U) && (a.hi == 0U);
}

static inline uint eq96(const cl_uint96_t a, const cl_uint96_t b)
{
	return (a.lo == b.lo) && (a.mid == b.mid) && (a.hi == b.hi);
}

static inline uint lt96(const cl_uint96_t a, const cl_uint96_t b)
{
	return (a.hi < b.hi) || ((a.hi == b.hi) && ((a.mid < b.mid) || ((a.mid == b.mid) && (a.lo < b.lo))));
}

static inline uint ge96(const cl_uint96_t a, const cl_uint96_t b)
{
	return !lt96(a, b);
}

static inline uint le96(const cl_uint96_t a, const cl_uint96_t b)
{
	return !lt96(b, a);
}

static inline cl_uint96_t add96_carry(const cl_uint96_t a, const cl_uint96_t b, __private uint *carry)
{
	cl_uint96_t r;
	uint c;

#ifdef __NV_CL_C_VERSION

	uint r0, r1, r2;

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2;\n\t"
		".reg .u32 bb0, bb1, bb2;\n\t"

		"mov.u32 aa0, %4;\n\t"
		"mov.u32 aa1, %5;\n\t"
		"mov.u32 aa2, %6;\n\t"
		"mov.u32 bb0, %7;\n\t"
		"mov.u32 bb1, %8;\n\t"
		"mov.u32 bb2, %9;\n\t"

		"add.cc.u32      %0, aa0, bb0;\n\t"
		"addc.cc.u32     %1, aa1, bb1;\n\t"
		"addc.cc.u32     %2, aa2, bb2;\n\t"
		"addc.u32        %3, 0, 0;\n\t"
		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(c)
		: "r"(a.lo), "r"(a.mid), "r"(a.hi),
		  "r"(b.lo), "r"(b.mid), "r"(b.hi)
	);

	r.lo = r0;
	r.mid = r1;
	r.hi = r2;

#else

	r.lo = a.lo + b.lo;
	const uint c0 = (r.lo < a.lo);

	const uint mid0 = a.mid + b.mid;
	const uint c1a = (mid0 < a.mid);
	r.mid = mid0 + c0;
	const uint c1b = (r.mid < mid0);
	const uint c1 = c1a | c1b;

	const uint hi0 = a.hi + b.hi;
	const uint c2a = (hi0 < a.hi);
	r.hi = hi0 + c1;
	const uint c2b = (r.hi < hi0);

	c = c2a | c2b;

#endif

	*carry = c;
	return r;
}

static inline cl_uint96_t add96_wrap(const cl_uint96_t a, const cl_uint96_t b)
{
	uint c;
	return add96_carry(a, b, &c);
}

static inline cl_uint96_t add_u32_carry(const cl_uint96_t a, const uint b, __private uint *carry)
{
	cl_uint96_t r;
	uint c;

#ifdef __NV_CL_C_VERSION

	uint r0, r1, r2;

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2;\n\t"
		"mov.u32 aa0, %4;\n\t"
		"mov.u32 aa1, %5;\n\t"
		"mov.u32 aa2, %6;\n\t"
		"add.cc.u32      %0, aa0, %7;\n\t"
		"addc.cc.u32     %1, aa1, 0;\n\t"
		"addc.cc.u32     %2, aa2, 0;\n\t"
		"addc.u32        %3, 0, 0;\n\t"
		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(c)
		: "r"(a.lo), "r"(a.mid), "r"(a.hi), "r"(b)
	);

	r.lo = r0;
	r.mid = r1;
	r.hi = r2;

#else

	r.lo = a.lo + b;
	const uint c0 = (r.lo < a.lo);

	r.mid = a.mid + c0;
	const uint c1 = (r.mid < a.mid);

	r.hi = a.hi + c1;
	c = (r.hi < a.hi);

#endif

	*carry = c;
	return r;
}

static inline cl_uint96_t add_u32_wrap(const cl_uint96_t a, const uint b)
{
	uint c;
	return add_u32_carry(a, b, &c);
}

static inline cl_uint96_t sub96_borrow(const cl_uint96_t a, const cl_uint96_t b, __private uint *borrow)
{
	cl_uint96_t r;
	uint br;

#ifdef __NV_CL_C_VERSION

	uint r0, r1, r2;

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2;\n\t"
		".reg .u32 bb0, bb1, bb2;\n\t"

		"mov.u32 aa0, %4;\n\t"
		"mov.u32 aa1, %5;\n\t"
		"mov.u32 aa2, %6;\n\t"
		"mov.u32 bb0, %7;\n\t"
		"mov.u32 bb1, %8;\n\t"
		"mov.u32 bb2, %9;\n\t"

		"sub.cc.u32      %0, aa0, bb0;\n\t"
		"subc.cc.u32     %1, aa1, bb1;\n\t"
		"subc.cc.u32     %2, aa2, bb2;\n\t"
		"subc.u32        %3, 0, 0;\n\t"
		"and.b32         %3, %3, 1;\n\t"
		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(br)
		: "r"(a.lo), "r"(a.mid), "r"(a.hi),
		  "r"(b.lo), "r"(b.mid), "r"(b.hi)
	);

	r.lo = r0;
	r.mid = r1;
	r.hi = r2;

#else

	const uint b0 = (a.lo < b.lo);
	r.lo = a.lo - b.lo;

	const uint mid0 = a.mid - b.mid;
	const uint b1a = (a.mid < b.mid);
	r.mid = mid0 - b0;
	const uint b1b = (mid0 < b0);
	const uint b1 = b1a | b1b;

	const uint hi0 = a.hi - b.hi;
	const uint b2a = (a.hi < b.hi);
	r.hi = hi0 - b1;
	const uint b2b = (hi0 < b1);

	br = b2a | b2b;

#endif

	*borrow = br;
	return r;
}

static inline cl_uint96_t sub96_wrap(const cl_uint96_t a, const cl_uint96_t b)
{
	uint borrow;
	return sub96_borrow(a, b, &borrow);
}

/* r = x + y (mod p), where 0 <= x,y < p */
static inline cl_uint96_t add_mod96(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p)
{
	const cl_uint96_t pmy = sub96_wrap(p, y);
	const uint reduce = ge96(x, pmy);
	cl_uint96_t r = add96_wrap(x, y);
	if (reduce) r = sub96_wrap(r, p);
	return r;
}

/* r = x + y (mod p), carry is the p-radix carry from x+y */
static inline cl_uint96_t add_mod_c96(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p, __private uint *carry)
{
	const cl_uint96_t pmy = sub96_wrap(p, y);
	const uint reduce = ge96(x, pmy);
	cl_uint96_t r = add96_wrap(x, y);
	if (reduce) r = sub96_wrap(r, p);
	*carry = reduce;
	return r;
}

/* Requires 0 <= x < p and y < p. */
static inline cl_uint96_t add_u32_mod96(const cl_uint96_t x, const uint y, const cl_uint96_t p)
{
	uint carry;
	cl_uint96_t r = add_u32_carry(x, y, &carry);
	if(carry || ge96(r, p))	r = sub96_wrap(r, p);
	return r;
}

/* r = x - y (mod p), where 0 <= x,y < p */
static inline cl_uint96_t sub_mod96(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p)
{
	uint borrow;
	cl_uint96_t r = sub96_borrow(x, y, &borrow);
	if (borrow) r = add96_wrap(r, p);
	return r;
}

/* r = x - y (mod p), borrow is the p-radix borrow from x-y */
static inline cl_uint96_t sub_mod_c96(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p, __private uint *carry)
{
	uint borrow;
	cl_uint96_t r = sub96_borrow(x, y, &borrow);
	if (borrow) r = add96_wrap(r, p);
	*carry = borrow;
	return r;
}

/* Allows y == p.  Since y == p is zero modulo p, x - y == x. */
static inline cl_uint96_t sub_mod96_y_le_p(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p)
{
	if (eq96(y, p)) return x;
	return sub_mod96(x, y, p);
}

static inline cl_uint96_t add_u32_allow_p(const cl_uint96_t x, const uint y)
{
	return add_u32_wrap(x, y);
}

static inline cl_uint192_t add192_carry(const cl_uint192_t a, const cl_uint192_t b, __private uint *carry)
{
	cl_uint192_t r;
	uint c;

#ifdef __NV_CL_C_VERSION

	uint r0, r1, r2, r3, r4, r5;

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2, aa3, aa4, aa5;\n\t"
		".reg .u32 bb0, bb1, bb2, bb3, bb4, bb5;\n\t"

		"mov.u32 aa0, %7;\n\t"
		"mov.u32 aa1, %8;\n\t"
		"mov.u32 aa2, %9;\n\t"
		"mov.u32 aa3, %10;\n\t"
		"mov.u32 aa4, %11;\n\t"
		"mov.u32 aa5, %12;\n\t"
		"mov.u32 bb0, %13;\n\t"
		"mov.u32 bb1, %14;\n\t"
		"mov.u32 bb2, %15;\n\t"
		"mov.u32 bb3, %16;\n\t"
		"mov.u32 bb4, %17;\n\t"
		"mov.u32 bb5, %18;\n\t"

		"add.cc.u32      %0, aa0, bb0;\n\t"
		"addc.cc.u32     %1, aa1, bb1;\n\t"
		"addc.cc.u32     %2, aa2, bb2;\n\t"
		"addc.cc.u32     %3, aa3, bb3;\n\t"
		"addc.cc.u32     %4, aa4, bb4;\n\t"
		"addc.cc.u32     %5, aa5, bb5;\n\t"
		"addc.u32        %6, 0, 0;\n\t"
		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2),
		  "=r"(r3), "=r"(r4), "=r"(r5), "=r"(c)
		: "r"(a.lo.lo), "r"(a.lo.mid), "r"(a.lo.hi),
		  "r"(a.hi.lo), "r"(a.hi.mid), "r"(a.hi.hi),
		  "r"(b.lo.lo), "r"(b.lo.mid), "r"(b.lo.hi),
		  "r"(b.hi.lo), "r"(b.hi.mid), "r"(b.hi.hi)
	);

	r.lo.lo = r0;
	r.lo.mid = r1;
	r.lo.hi = r2;
	r.hi.lo = r3;
	r.hi.mid = r4;
	r.hi.hi = r5;

#else

	uint c0, c1, c2;

	r.lo = add96_carry(a.lo, b.lo, &c0);
	r.hi = add96_carry(a.hi, b.hi, &c1);

	if (c0)
		r.hi = add96_carry(r.hi, one96(), &c2);
	else
		c2 = 0U;

	c = c1 | c2;

#endif

	*carry = c;
	return r;
}

static inline cl_uint192_t add192_wrap(const cl_uint192_t a, const cl_uint192_t b)
{
	uint c;
	return add192_carry(a, b, &c);
}

static inline cl_uint192_t wide_from_low96(const cl_uint96_t lo)
{
	cl_uint192_t r;
	r.lo = lo;
	r.hi = zero96();
	return r;
}

static inline cl_uint96_t mul_lo96(
	const cl_uint96_t a,
	const cl_uint96_t b)
{
	const uint a0 = a.lo;
	const uint a1 = a.mid;
	const uint a2 = a.hi;

	const uint b0 = b.lo;
	const uint b1 = b.mid;
	const uint b2 = b.hi;

	uint r0, r1, r2;

#ifdef __NV_CL_C_VERSION

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2;\n\t"
		".reg .u32 bb0, bb1, bb2;\n\t"

		"mov.u32 aa0, %3;\n\t"
		"mov.u32 aa1, %4;\n\t"
		"mov.u32 aa2, %5;\n\t"
		"mov.u32 bb0, %6;\n\t"
		"mov.u32 bb1, %7;\n\t"
		"mov.u32 bb2, %8;\n\t"

		/*
			Column 0:
				r0 = low(a0*b0)
				r1 = high(a0*b0)
		*/
		"mul.lo.u32       %0, aa0, bb0;\n\t"
		"mul.hi.u32       %1, aa0, bb0;\n\t"

		/*
			Column 1:
				r1 += low(a0*b1)
				r2  = high(a0*b1) + carry

				r1 += low(a1*b0)
				r2 += high(a1*b0) + carry

			Carry out of r2 is word 3, which is discarded for low96.
		*/
		"mad.lo.cc.u32    %1, aa0, bb1, %1;\n\t"
		"madc.hi.u32      %2, aa0, bb1, 0;\n\t"

		"mad.lo.cc.u32    %1, aa1, bb0, %1;\n\t"
		"madc.hi.u32      %2, aa1, bb0, %2;\n\t"

		/*
			Column 2:
				Only low 32 bits of these products affect r2.
				Carries out of r2 are discarded mod 2^96.
		*/
		"mad.lo.u32       %2, aa0, bb2, %2;\n\t"
		"mad.lo.u32       %2, aa1, bb1, %2;\n\t"
		"mad.lo.u32       %2, aa2, bb0, %2;\n\t"

		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2)
		: "r"(a0), "r"(a1), "r"(a2),
		  "r"(b0), "r"(b1), "r"(b2)
	);

#else

	ulong carry;
	ulong sum_lo;
	ulong sum_hi;

#define ADD_PROD_32(x_, y_)                    \
	do {                                       \
		const uint _x = (x_);                  \
		const uint _y = (y_);                  \
		sum_lo += (ulong)(_x * _y);            \
		sum_hi += (ulong)mul_hi(_x, _y);       \
	} while (0)

#define ADD_PROD_LO32(x_, y_)                  \
	do {                                       \
		const uint _x = (x_);                  \
		const uint _y = (y_);                  \
		sum_lo += (ulong)(_x * _y);            \
	} while (0)

	// Column 0
	carry = 0;
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a0, b0);
	r0 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 1
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a0, b1);
	ADD_PROD_32(a1, b0);
	r1 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 2, low half only
	sum_lo = carry & 0xffffffffUL;
	ADD_PROD_LO32(a0, b2);
	ADD_PROD_LO32(a1, b1);
	ADD_PROD_LO32(a2, b0);
	r2 = (uint)sum_lo;

#undef ADD_PROD_LO32
#undef ADD_PROD_32

#endif

	cl_uint96_t r;
	r.lo = r0;
	r.mid = r1;
	r.hi = r2;
	return r;
}

static inline cl_uint192_t mul_wide96(
	const cl_uint96_t a,
	const cl_uint96_t b)
{
	const uint a0 = a.lo;
	const uint a1 = a.mid;
	const uint a2 = a.hi;

	const uint b0 = b.lo;
	const uint b1 = b.mid;
	const uint b2 = b.hi;

	uint r0, r1, r2, r3, r4, r5;

#ifdef __NV_CL_C_VERSION

	asm volatile(
		"{\n\t"
		".reg .u32 aa0, aa1, aa2;\n\t"
		".reg .u32 bb0, bb1, bb2;\n\t"
		".reg .u32 c0, c1;\n\t"

		"mov.u32 aa0, %6;\n\t"
		"mov.u32 aa1, %7;\n\t"
		"mov.u32 aa2, %8;\n\t"
		"mov.u32 bb0, %9;\n\t"
		"mov.u32 bb1, %10;\n\t"
		"mov.u32 bb2, %11;\n\t"

		/*
			Column 0:
				total = a0*b0
				r0 = low32(total)
				carry = total >> 32
		*/
		"mul.lo.u32       %0, aa0, bb0;\n\t"
		"mul.hi.u32       c0, aa0, bb0;\n\t"
		"mov.u32          c1, 0;\n\t"

		/*
			Column 1:
				total = carry + a0*b1 + a1*b0
		*/
		"mov.u32          %1, c0;\n\t"
		"mov.u32          c0, c1;\n\t"
		"mov.u32          c1, 0;\n\t"

		"mad.lo.cc.u32    %1, aa0, bb1, %1;\n\t"
		"madc.hi.cc.u32   c0, aa0, bb1, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		"mad.lo.cc.u32    %1, aa1, bb0, %1;\n\t"
		"madc.hi.cc.u32   c0, aa1, bb0, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		/*
			Column 2:
				total = carry + a0*b2 + a1*b1 + a2*b0
		*/
		"mov.u32          %2, c0;\n\t"
		"mov.u32          c0, c1;\n\t"
		"mov.u32          c1, 0;\n\t"

		"mad.lo.cc.u32    %2, aa0, bb2, %2;\n\t"
		"madc.hi.cc.u32   c0, aa0, bb2, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		"mad.lo.cc.u32    %2, aa1, bb1, %2;\n\t"
		"madc.hi.cc.u32   c0, aa1, bb1, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		"mad.lo.cc.u32    %2, aa2, bb0, %2;\n\t"
		"madc.hi.cc.u32   c0, aa2, bb0, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		/*
			Column 3:
				total = carry + a1*b2 + a2*b1
		*/
		"mov.u32          %3, c0;\n\t"
		"mov.u32          c0, c1;\n\t"
		"mov.u32          c1, 0;\n\t"

		"mad.lo.cc.u32    %3, aa1, bb2, %3;\n\t"
		"madc.hi.cc.u32   c0, aa1, bb2, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		"mad.lo.cc.u32    %3, aa2, bb1, %3;\n\t"
		"madc.hi.cc.u32   c0, aa2, bb1, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		/*
			Column 4:
				total = carry + a2*b2
		*/
		"mov.u32          %4, c0;\n\t"
		"mov.u32          c0, c1;\n\t"
		"mov.u32          c1, 0;\n\t"

		"mad.lo.cc.u32    %4, aa2, bb2, %4;\n\t"
		"madc.hi.cc.u32   c0, aa2, bb2, c0;\n\t"
		"addc.u32         c1, c1, 0;\n\t"

		/*
			Column 5:
				final carry.
			For a true 96x96 product, c1 must be zero here.
		*/
		"mov.u32          %5, c0;\n\t"

		"}\n\t"
		: "=r"(r0), "=r"(r1), "=r"(r2),
		  "=r"(r3), "=r"(r4), "=r"(r5)
		: "r"(a0), "r"(a1), "r"(a2),
		  "r"(b0), "r"(b1), "r"(b2)
	);

#else

	ulong carry;
	ulong sum_lo;
	ulong sum_hi;

#define ADD_PROD_32(x_, y_)                    \
	do {                                       \
		const uint _x = (x_);                  \
		const uint _y = (y_);                  \
		sum_lo += (ulong)(_x * _y);            \
		sum_hi += (ulong)mul_hi(_x, _y);       \
	} while (0)

	carry = 0;

	// Column 0
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a0, b0);
	r0 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 1
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a0, b1);
	ADD_PROD_32(a1, b0);
	r1 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 2
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a0, b2);
	ADD_PROD_32(a1, b1);
	ADD_PROD_32(a2, b0);
	r2 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 3
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a1, b2);
	ADD_PROD_32(a2, b1);
	r3 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 4
	sum_lo = carry & 0xffffffffUL;
	sum_hi = carry >> 32;
	ADD_PROD_32(a2, b2);
	r4 = (uint)sum_lo;
	carry = sum_hi + (sum_lo >> 32);

	// Column 5
	r5 = (uint)carry;

#undef ADD_PROD_32

#endif

	cl_uint192_t r;

	r.lo.lo = r0;
	r.lo.mid = r1;
	r.lo.hi = r2;

	r.hi.lo = r3;
	r.hi.mid = r4;
	r.hi.hi = r5;

	return r;
}

/*
	r0 + p*r1 = 2 * (x0 + p*x1) mod p^2, with 0 <= r0,r1 < p.
*/
static inline m2p96_t m2p_dup96(const m2p96_t x, const cl_uint96_t p)
{
	m2p96_t r;
	uint c;

	r.r0 = add_mod_c96(x.r0, x.r0, p, &c);
	r.r1 = add_mod96(x.r1, x.r1, p);
	if (c) r.r1 = add_u32_mod96(r.r1, 1U, p);

	return r;
}

/*
	R mod p^2, where R = 2^96, represented as r0 + p*r1.

	This avoids 96-bit division in the kernel.  
*/
static inline m2p96_t m2p_one96(const cl_uint96_t p)
{
	m2p96_t x;

	if(!p.hi){
		// For p < 2^64, keep the generic path.
		x.r0 = one96();
		x.r1 = zero96();

		for(int i = 0; i < 96; i++)
			x = m2p_dup96(x, p);

		return x;
	}

	// p > 2^64 path.
	// Start with r0 = 2^64, r1 = 0.
	cl_uint96_t r0 = make96_3(0U, 0U, 1U);
	uint r1 = 0U;

	for(int i = 0; i < 32; i++){
		uint carry96;
		// r0 = 2*r0 mod 2^96, carry96 is the overflow bit above 96 bits.
		r0 = add96_carry(r0, r0, &carry96);
		uint carryp = carry96 || ge96(r0, p);
		if(carryp) r0 = sub96_wrap(r0, p);
		// r1 must also be doubled.
		r1 = (r1 << 1) | carryp;
	}

	x.r0 = r0;
	x.r1 = make96_3(r1, 0U, 0U);

	return x;
}

/*
	r0 + p*r1 = x*y*R^(-1) mod p^2, with 0 <= x,y < p.
	This is the special x1 = y1 = 0 path from the original kernel.
*/
static inline m2p96_t m2p_mul_s96(const cl_uint96_t x, const cl_uint96_t y, const cl_uint96_t p, const cl_uint96_t q)
{
	const cl_uint192_t t = mul_wide96(x, y);
	const cl_uint96_t u0 = mul_lo96(q, t.lo);

	const cl_uint96_t t1 = t.hi;
	const cl_uint96_t v1 = mul_wide96(p, u0).hi;

	const cl_uint96_t qu0 = mul_lo96(q, u0);
	const cl_uint96_t v1p = mul_wide96(p, qu0).hi;

	uint c;
	m2p96_t z;
	z.r0 = sub_mod_c96(t1, v1, p, &c);

	cl_uint96_t y1 = v1p;
	if (c) y1 = add_u32_allow_p(y1, 1U);   // may become exactly p
	z.r1 = sub_mod96_y_le_p(zero96(), y1, p);

	return z;
}

/*
	r0 + p*r1 = (x0 + p*x1)^2 * R^(-1) mod p^2, with 0 <= x0,x1 < p.
*/
static inline m2p96_t m2p_square96(const m2p96_t x, const cl_uint96_t p, const cl_uint96_t q)
{
	const cl_uint192_t t = mul_wide96(x.r0, x.r0);
	const cl_uint96_t u0 = mul_lo96(q, t.lo);

	const cl_uint96_t t1 = t.hi;
	const cl_uint96_t v1 = mul_wide96(p, u0).hi;

	const cl_uint192_t x01 = mul_wide96(x.r0, x.r1);
	const cl_uint192_t u0w = wide_from_low96(u0);

	cl_uint192_t x01u = add192_wrap(x01, u0w);

	uint tp_carry;
	const cl_uint192_t tp = add192_carry(x01u, x01, &tp_carry);

	cl_uint96_t t1p = tp.hi;
	if (tp_carry || ge96(t1p, p))
		t1p = sub96_wrap(t1p, p);

	const cl_uint96_t up0 = mul_lo96(q, tp.lo);
	const cl_uint96_t v1p = mul_wide96(p, up0).hi;

	uint c;
	m2p96_t z;
	z.r0 = sub_mod_c96(t1, v1, p, &c);

	cl_uint96_t y1 = v1p;
	if (c) y1 = add_u32_allow_p(y1, 1U);   // may become exactly p
	z.r1 = sub_mod96_y_le_p(t1p, y1, p);

	return z;
}

/* Convert a Montgomery residue modulo p^2 back to the integer residue. */
static inline m2p96_t m2p_get96(const m2p96_t x, const cl_uint96_t p, const cl_uint96_t q)
{
	const cl_uint96_t u0 = mul_lo96(q, x.r0);
	const cl_uint96_t v1 = mul_wide96(p, u0).hi;

	uint tp_carry;
	const cl_uint96_t tp = add96_carry(x.r1, u0, &tp_carry);

	const cl_uint96_t up0 = mul_lo96(q, tp);
	const cl_uint96_t t1p = make96_3(tp_carry, 0U, 0U);
	const cl_uint96_t v1p = mul_wide96(p, up0).hi;

	uint c;
	m2p96_t z;
	z.r0 = sub_mod_c96(zero96(), v1, p, &c);

	cl_uint96_t y1 = v1p;
	if (c) y1 = add_u32_allow_p(y1, 1U);   // may become exactly p
	z.r1 = sub_mod96_y_le_p(t1p, y1, p);

	return z;
}

static inline int msb_index96(const cl_uint96_t x)
{
	if (x.hi != 0U)
		return 64 + 31 - (int)clz(x.hi);

	if (x.mid != 0U)
		return 32 + 31 - (int)clz(x.mid);

	return 31 - (int)clz(x.lo);
}

static inline uint bit_test96(const cl_uint96_t x, const int bit)
{
	if (bit < 32)
		return (x.lo >> bit) & 1U;

	if (bit < 64)
		return (x.mid >> (bit - 32)) & 1U;

	return (x.hi >> (bit - 64)) & 1U;
}

static inline cl_uint96_t shr1_96(const cl_uint96_t x)
{
	cl_uint96_t r;
	r.lo = (x.lo >> 1) | (x.mid << 31);
	r.mid = (x.mid >> 1) | ((x.hi & 1U) << 31);
	r.hi = x.hi >> 1;
	return r;
}

static inline int small96_to_int(const cl_uint96_t x)
{
	return (int)x.lo;
}

/* 2^e mod p^2 in Montgomery/p-radix form */
static inline m2p96_t pow2_m2p96(const cl_uint96_t e, const m2p96_t one, const cl_uint96_t p, const cl_uint96_t q)
{
	// e must be greater than 2
	int bit = msb_index96(e) - 1;

	m2p96_t a = m2p_dup96(one, p);

	while ((bit >= 0) && is_zero96(a.r1))
	{
		a = m2p_mul_s96(a.r0, a.r0, p, q);
		if (bit_test96(e, bit))
			a = m2p_dup96(a, p);
		bit--;
	}

	while (bit >= 0)
	{
		a = m2p_square96(a, p, q);
		if (bit_test96(e, bit))
			a = m2p_dup96(a, p);
		bit--;
	}

	return a;
}

/* Convert one Montgomery residue modulo p back to ordinary form. */
static inline cl_uint96_t mont_get_modp96(
	const cl_uint96_t x,
	const cl_uint96_t p,
	const cl_uint96_t q)
{
	const cl_uint96_t u0 = mul_lo96(q, x);
	const cl_uint96_t v1 = mul_wide96(p, u0).hi;

	return sub_mod96(zero96(), v1, p);
}

static inline uint inv32_odd(const uint a)
{
	/*
		Inverse of odd a modulo 2^32.

		Starting from 1 gives one correct bit.
		Each Newton step doubles the number of correct bits.
	*/
	uint x = 1U;

	x *= 2U - a * x;   // 2 bits
	x *= 2U - a * x;   // 4 bits
	x *= 2U - a * x;   // 8 bits
	x *= 2U - a * x;   // 16 bits
	x *= 2U - a * x;   // 32 bits

	return x;
}

/*
	Positive Montgomery inverse:

		q = p^(-1) mod 2^96

	This replaces the 7-step 96-bit Newton loop.
*/

static inline cl_uint96_t mont_inv_pos96(const cl_uint96_t p)
{
	const uint p0 = p.lo;
	const uint p1 = p.mid;
	const uint p2 = p.hi;

	const uint q0 = inv32_odd(p0);

#if defined(__NV_CL_C_VERSION)

	uint q1, q2;

	asm volatile(
		"{\n\t"
		".reg .u32 carry0;\n\t"
		".reg .u32 p1q0_lo;\n\t"
		".reg .u32 p1q0_hi;\n\t"
		".reg .u32 s1;\n\t"
		".reg .u32 q1r;\n\t"
		".reg .u32 q2r;\n\t"
		".reg .u32 p0q1_lo;\n\t"
		".reg .u32 p0q1_hi;\n\t"
		".reg .u32 lowtmp;\n\t"
		".reg .u32 carry1;\n\t"
		".reg .u32 s2;\n\t"

		// carry0 = high32(p0*q0)
		"mul.hi.u32 carry0, %2, %4;\n\t"

		// p1*q0
		"mul.lo.u32 p1q0_lo, %3, %4;\n\t"
		"mul.hi.u32 p1q0_hi, %3, %4;\n\t"

		// s1 = low32(p1*q0 + carry0)
		"add.u32 s1, p1q0_lo, carry0;\n\t"

		// q1 = -low32(s1*q0)
		"mul.lo.u32 q1r, s1, %4;\n\t"
		"sub.u32 q1r, 0, q1r;\n\t"

		// p0*q1
		"mul.lo.u32 p0q1_lo, %2, q1r;\n\t"
		"mul.hi.u32 p0q1_hi, %2, q1r;\n\t"

		/*
			carry1 = high32(p0*q1 + p1*q0 + carry0)

			Overflow above 64 bits is intentionally discarded.
		*/
		"add.cc.u32 lowtmp, p0q1_lo, p1q0_lo;\n\t"
		"addc.u32 carry1, p0q1_hi, p1q0_hi;\n\t"
		"add.cc.u32 lowtmp, lowtmp, carry0;\n\t"
		"addc.u32 carry1, carry1, 0;\n\t"

		// s2 = low32(p1*q1 + p2*q0 + carry1)
		"mad.lo.u32 s2, %5, %4, carry1;\n\t"
		"mad.lo.u32 s2, %3, q1r, s2;\n\t"

		// q2 = -low32(s2*q0)
		"mul.lo.u32 q2r, s2, %4;\n\t"
		"sub.u32 q2r, 0, q2r;\n\t"

		"mov.u32 %0, q1r;\n\t"
		"mov.u32 %1, q2r;\n\t"
		"}"
		: "=r"(q1), "=r"(q2)
		: "r"(p0), "r"(p1), "r"(q0), "r"(p2)
	);

	return make96_3(q0, q1, q2);

#else

	const uint carry0 = mul_hi(p0, q0);

	const uint p1q0_lo = p1 * q0;
	const uint p1q0_hi = mul_hi(p1, q0);

	const uint s1 = p1q0_lo + carry0;
	const uint q1 = 0U - (s1 * q0);

	const uint p0q1_lo = p0 * q1;
	const uint p0q1_hi = mul_hi(p0, q1);

	uint lowtmp = p0q1_lo + p1q0_lo;
	uint c = (lowtmp < p0q1_lo);

	const uint old_lowtmp = lowtmp;
	lowtmp += carry0;
	c += (lowtmp < old_lowtmp);

	const uint carry1 = p0q1_hi + p1q0_hi + c;

	const uint s2 = (p1 * q1) + (p2 * q0) + carry1;
	const uint q2 = 0U - (s2 * q0);

	return make96_3(q0, q1, q2);

#endif
}
