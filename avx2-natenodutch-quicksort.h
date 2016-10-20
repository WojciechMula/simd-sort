#pragma once

#include <x86intrin.h>
#include <cstdint>
#include <memory.h>
#include "avx2-altquicksort.h"
/**
* This is an alternative approach to SIMD quicksort, based on a description by
* Nathan Kurz. Daniel Lemire implemented it to avoid the  Dutch national flag
* problem as an experiment. This is not meant to be fast, as implemented. It is
* an experiment.
*/

/*
I'm going to use "vector" as a unit, but it doesn't have to be the
native vector size.  In particular, I think we can read multiple
vector register's worth of data with minimal added time per loop
iteration.

First, read two vectors from the head of the list, and two from the
tail of the list.   Store these, either in registers or written to a
buffer.  We're not going to use these until the end.  We're just
getting them out of the way so that we have more than one vectors
writeable space at the head and tail.

wh--rh------------------rt--wt-

wh = write head
rh = read head
rt = read tail
wt = write tail

Then read one vector from head, and advance rh accordingly.  Call it nextVector.

Loop:

Set currentVector = nextVector.

Set nextVectorPtr to either rh or rt depending on which has the least
"writable space" available.  That is, if (rh - wh) < (wt - rt), read
from and advance rh, else read from and update rt.   These updates
need to be branchless.

Read nextVector = *nextVectorPtr.

Partition currentVector (lookup, shuffle) and write to both wh and wt
(tail offset so end hits wt).
Advance wh and wt based on number of valid entries.  Use a second byte
lookup issued as early as possible to determine this count.  Our
performance is dependent on  the latency of this step.

-----
*/
static uint32_t shufflemask[256 * 8] __attribute__((aligned(0x100))) = {
    0, 1, 2, 3, 4, 5, 6, 7, /* 0*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 1*/
    1, 0, 2, 3, 4, 5, 6, 7, /* 2*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 3*/
    2, 0, 1, 3, 4, 5, 6, 7, /* 4*/
    0, 2, 1, 3, 4, 5, 6, 7, /* 5*/
    1, 2, 0, 3, 4, 5, 6, 7, /* 6*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 7*/
    3, 0, 1, 2, 4, 5, 6, 7, /* 8*/
    0, 3, 1, 2, 4, 5, 6, 7, /* 9*/
    1, 3, 0, 2, 4, 5, 6, 7, /* 10*/
    0, 1, 3, 2, 4, 5, 6, 7, /* 11*/
    2, 3, 0, 1, 4, 5, 6, 7, /* 12*/
    0, 2, 3, 1, 4, 5, 6, 7, /* 13*/
    1, 2, 3, 0, 4, 5, 6, 7, /* 14*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 15*/
    4, 0, 1, 2, 3, 5, 6, 7, /* 16*/
    0, 4, 1, 2, 3, 5, 6, 7, /* 17*/
    1, 4, 0, 2, 3, 5, 6, 7, /* 18*/
    0, 1, 4, 2, 3, 5, 6, 7, /* 19*/
    2, 4, 0, 1, 3, 5, 6, 7, /* 20*/
    0, 2, 4, 1, 3, 5, 6, 7, /* 21*/
    1, 2, 4, 0, 3, 5, 6, 7, /* 22*/
    0, 1, 2, 4, 3, 5, 6, 7, /* 23*/
    3, 4, 0, 1, 2, 5, 6, 7, /* 24*/
    0, 3, 4, 1, 2, 5, 6, 7, /* 25*/
    1, 3, 4, 0, 2, 5, 6, 7, /* 26*/
    0, 1, 3, 4, 2, 5, 6, 7, /* 27*/
    2, 3, 4, 0, 1, 5, 6, 7, /* 28*/
    0, 2, 3, 4, 1, 5, 6, 7, /* 29*/
    1, 2, 3, 4, 0, 5, 6, 7, /* 30*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 31*/
    5, 0, 1, 2, 3, 4, 6, 7, /* 32*/
    0, 5, 1, 2, 3, 4, 6, 7, /* 33*/
    1, 5, 0, 2, 3, 4, 6, 7, /* 34*/
    0, 1, 5, 2, 3, 4, 6, 7, /* 35*/
    2, 5, 0, 1, 3, 4, 6, 7, /* 36*/
    0, 2, 5, 1, 3, 4, 6, 7, /* 37*/
    1, 2, 5, 0, 3, 4, 6, 7, /* 38*/
    0, 1, 2, 5, 3, 4, 6, 7, /* 39*/
    3, 5, 0, 1, 2, 4, 6, 7, /* 40*/
    0, 3, 5, 1, 2, 4, 6, 7, /* 41*/
    1, 3, 5, 0, 2, 4, 6, 7, /* 42*/
    0, 1, 3, 5, 2, 4, 6, 7, /* 43*/
    2, 3, 5, 0, 1, 4, 6, 7, /* 44*/
    0, 2, 3, 5, 1, 4, 6, 7, /* 45*/
    1, 2, 3, 5, 0, 4, 6, 7, /* 46*/
    0, 1, 2, 3, 5, 4, 6, 7, /* 47*/
    4, 5, 0, 1, 2, 3, 6, 7, /* 48*/
    0, 4, 5, 1, 2, 3, 6, 7, /* 49*/
    1, 4, 5, 0, 2, 3, 6, 7, /* 50*/
    0, 1, 4, 5, 2, 3, 6, 7, /* 51*/
    2, 4, 5, 0, 1, 3, 6, 7, /* 52*/
    0, 2, 4, 5, 1, 3, 6, 7, /* 53*/
    1, 2, 4, 5, 0, 3, 6, 7, /* 54*/
    0, 1, 2, 4, 5, 3, 6, 7, /* 55*/
    3, 4, 5, 0, 1, 2, 6, 7, /* 56*/
    0, 3, 4, 5, 1, 2, 6, 7, /* 57*/
    1, 3, 4, 5, 0, 2, 6, 7, /* 58*/
    0, 1, 3, 4, 5, 2, 6, 7, /* 59*/
    2, 3, 4, 5, 0, 1, 6, 7, /* 60*/
    0, 2, 3, 4, 5, 1, 6, 7, /* 61*/
    1, 2, 3, 4, 5, 0, 6, 7, /* 62*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 63*/
    6, 0, 1, 2, 3, 4, 5, 7, /* 64*/
    0, 6, 1, 2, 3, 4, 5, 7, /* 65*/
    1, 6, 0, 2, 3, 4, 5, 7, /* 66*/
    0, 1, 6, 2, 3, 4, 5, 7, /* 67*/
    2, 6, 0, 1, 3, 4, 5, 7, /* 68*/
    0, 2, 6, 1, 3, 4, 5, 7, /* 69*/
    1, 2, 6, 0, 3, 4, 5, 7, /* 70*/
    0, 1, 2, 6, 3, 4, 5, 7, /* 71*/
    3, 6, 0, 1, 2, 4, 5, 7, /* 72*/
    0, 3, 6, 1, 2, 4, 5, 7, /* 73*/
    1, 3, 6, 0, 2, 4, 5, 7, /* 74*/
    0, 1, 3, 6, 2, 4, 5, 7, /* 75*/
    2, 3, 6, 0, 1, 4, 5, 7, /* 76*/
    0, 2, 3, 6, 1, 4, 5, 7, /* 77*/
    1, 2, 3, 6, 0, 4, 5, 7, /* 78*/
    0, 1, 2, 3, 6, 4, 5, 7, /* 79*/
    4, 6, 0, 1, 2, 3, 5, 7, /* 80*/
    0, 4, 6, 1, 2, 3, 5, 7, /* 81*/
    1, 4, 6, 0, 2, 3, 5, 7, /* 82*/
    0, 1, 4, 6, 2, 3, 5, 7, /* 83*/
    2, 4, 6, 0, 1, 3, 5, 7, /* 84*/
    0, 2, 4, 6, 1, 3, 5, 7, /* 85*/
    1, 2, 4, 6, 0, 3, 5, 7, /* 86*/
    0, 1, 2, 4, 6, 3, 5, 7, /* 87*/
    3, 4, 6, 0, 1, 2, 5, 7, /* 88*/
    0, 3, 4, 6, 1, 2, 5, 7, /* 89*/
    1, 3, 4, 6, 0, 2, 5, 7, /* 90*/
    0, 1, 3, 4, 6, 2, 5, 7, /* 91*/
    2, 3, 4, 6, 0, 1, 5, 7, /* 92*/
    0, 2, 3, 4, 6, 1, 5, 7, /* 93*/
    1, 2, 3, 4, 6, 0, 5, 7, /* 94*/
    0, 1, 2, 3, 4, 6, 5, 7, /* 95*/
    5, 6, 0, 1, 2, 3, 4, 7, /* 96*/
    0, 5, 6, 1, 2, 3, 4, 7, /* 97*/
    1, 5, 6, 0, 2, 3, 4, 7, /* 98*/
    0, 1, 5, 6, 2, 3, 4, 7, /* 99*/
    2, 5, 6, 0, 1, 3, 4, 7, /* 100*/
    0, 2, 5, 6, 1, 3, 4, 7, /* 101*/
    1, 2, 5, 6, 0, 3, 4, 7, /* 102*/
    0, 1, 2, 5, 6, 3, 4, 7, /* 103*/
    3, 5, 6, 0, 1, 2, 4, 7, /* 104*/
    0, 3, 5, 6, 1, 2, 4, 7, /* 105*/
    1, 3, 5, 6, 0, 2, 4, 7, /* 106*/
    0, 1, 3, 5, 6, 2, 4, 7, /* 107*/
    2, 3, 5, 6, 0, 1, 4, 7, /* 108*/
    0, 2, 3, 5, 6, 1, 4, 7, /* 109*/
    1, 2, 3, 5, 6, 0, 4, 7, /* 110*/
    0, 1, 2, 3, 5, 6, 4, 7, /* 111*/
    4, 5, 6, 0, 1, 2, 3, 7, /* 112*/
    0, 4, 5, 6, 1, 2, 3, 7, /* 113*/
    1, 4, 5, 6, 0, 2, 3, 7, /* 114*/
    0, 1, 4, 5, 6, 2, 3, 7, /* 115*/
    2, 4, 5, 6, 0, 1, 3, 7, /* 116*/
    0, 2, 4, 5, 6, 1, 3, 7, /* 117*/
    1, 2, 4, 5, 6, 0, 3, 7, /* 118*/
    0, 1, 2, 4, 5, 6, 3, 7, /* 119*/
    3, 4, 5, 6, 0, 1, 2, 7, /* 120*/
    0, 3, 4, 5, 6, 1, 2, 7, /* 121*/
    1, 3, 4, 5, 6, 0, 2, 7, /* 122*/
    0, 1, 3, 4, 5, 6, 2, 7, /* 123*/
    2, 3, 4, 5, 6, 0, 1, 7, /* 124*/
    0, 2, 3, 4, 5, 6, 1, 7, /* 125*/
    1, 2, 3, 4, 5, 6, 0, 7, /* 126*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 127*/
    7, 0, 1, 2, 3, 4, 5, 6, /* 128*/
    0, 7, 1, 2, 3, 4, 5, 6, /* 129*/
    1, 7, 0, 2, 3, 4, 5, 6, /* 130*/
    0, 1, 7, 2, 3, 4, 5, 6, /* 131*/
    2, 7, 0, 1, 3, 4, 5, 6, /* 132*/
    0, 2, 7, 1, 3, 4, 5, 6, /* 133*/
    1, 2, 7, 0, 3, 4, 5, 6, /* 134*/
    0, 1, 2, 7, 3, 4, 5, 6, /* 135*/
    3, 7, 0, 1, 2, 4, 5, 6, /* 136*/
    0, 3, 7, 1, 2, 4, 5, 6, /* 137*/
    1, 3, 7, 0, 2, 4, 5, 6, /* 138*/
    0, 1, 3, 7, 2, 4, 5, 6, /* 139*/
    2, 3, 7, 0, 1, 4, 5, 6, /* 140*/
    0, 2, 3, 7, 1, 4, 5, 6, /* 141*/
    1, 2, 3, 7, 0, 4, 5, 6, /* 142*/
    0, 1, 2, 3, 7, 4, 5, 6, /* 143*/
    4, 7, 0, 1, 2, 3, 5, 6, /* 144*/
    0, 4, 7, 1, 2, 3, 5, 6, /* 145*/
    1, 4, 7, 0, 2, 3, 5, 6, /* 146*/
    0, 1, 4, 7, 2, 3, 5, 6, /* 147*/
    2, 4, 7, 0, 1, 3, 5, 6, /* 148*/
    0, 2, 4, 7, 1, 3, 5, 6, /* 149*/
    1, 2, 4, 7, 0, 3, 5, 6, /* 150*/
    0, 1, 2, 4, 7, 3, 5, 6, /* 151*/
    3, 4, 7, 0, 1, 2, 5, 6, /* 152*/
    0, 3, 4, 7, 1, 2, 5, 6, /* 153*/
    1, 3, 4, 7, 0, 2, 5, 6, /* 154*/
    0, 1, 3, 4, 7, 2, 5, 6, /* 155*/
    2, 3, 4, 7, 0, 1, 5, 6, /* 156*/
    0, 2, 3, 4, 7, 1, 5, 6, /* 157*/
    1, 2, 3, 4, 7, 0, 5, 6, /* 158*/
    0, 1, 2, 3, 4, 7, 5, 6, /* 159*/
    5, 7, 0, 1, 2, 3, 4, 6, /* 160*/
    0, 5, 7, 1, 2, 3, 4, 6, /* 161*/
    1, 5, 7, 0, 2, 3, 4, 6, /* 162*/
    0, 1, 5, 7, 2, 3, 4, 6, /* 163*/
    2, 5, 7, 0, 1, 3, 4, 6, /* 164*/
    0, 2, 5, 7, 1, 3, 4, 6, /* 165*/
    1, 2, 5, 7, 0, 3, 4, 6, /* 166*/
    0, 1, 2, 5, 7, 3, 4, 6, /* 167*/
    3, 5, 7, 0, 1, 2, 4, 6, /* 168*/
    0, 3, 5, 7, 1, 2, 4, 6, /* 169*/
    1, 3, 5, 7, 0, 2, 4, 6, /* 170*/
    0, 1, 3, 5, 7, 2, 4, 6, /* 171*/
    2, 3, 5, 7, 0, 1, 4, 6, /* 172*/
    0, 2, 3, 5, 7, 1, 4, 6, /* 173*/
    1, 2, 3, 5, 7, 0, 4, 6, /* 174*/
    0, 1, 2, 3, 5, 7, 4, 6, /* 175*/
    4, 5, 7, 0, 1, 2, 3, 6, /* 176*/
    0, 4, 5, 7, 1, 2, 3, 6, /* 177*/
    1, 4, 5, 7, 0, 2, 3, 6, /* 178*/
    0, 1, 4, 5, 7, 2, 3, 6, /* 179*/
    2, 4, 5, 7, 0, 1, 3, 6, /* 180*/
    0, 2, 4, 5, 7, 1, 3, 6, /* 181*/
    1, 2, 4, 5, 7, 0, 3, 6, /* 182*/
    0, 1, 2, 4, 5, 7, 3, 6, /* 183*/
    3, 4, 5, 7, 0, 1, 2, 6, /* 184*/
    0, 3, 4, 5, 7, 1, 2, 6, /* 185*/
    1, 3, 4, 5, 7, 0, 2, 6, /* 186*/
    0, 1, 3, 4, 5, 7, 2, 6, /* 187*/
    2, 3, 4, 5, 7, 0, 1, 6, /* 188*/
    0, 2, 3, 4, 5, 7, 1, 6, /* 189*/
    1, 2, 3, 4, 5, 7, 0, 6, /* 190*/
    0, 1, 2, 3, 4, 5, 7, 6, /* 191*/
    6, 7, 0, 1, 2, 3, 4, 5, /* 192*/
    0, 6, 7, 1, 2, 3, 4, 5, /* 193*/
    1, 6, 7, 0, 2, 3, 4, 5, /* 194*/
    0, 1, 6, 7, 2, 3, 4, 5, /* 195*/
    2, 6, 7, 0, 1, 3, 4, 5, /* 196*/
    0, 2, 6, 7, 1, 3, 4, 5, /* 197*/
    1, 2, 6, 7, 0, 3, 4, 5, /* 198*/
    0, 1, 2, 6, 7, 3, 4, 5, /* 199*/
    3, 6, 7, 0, 1, 2, 4, 5, /* 200*/
    0, 3, 6, 7, 1, 2, 4, 5, /* 201*/
    1, 3, 6, 7, 0, 2, 4, 5, /* 202*/
    0, 1, 3, 6, 7, 2, 4, 5, /* 203*/
    2, 3, 6, 7, 0, 1, 4, 5, /* 204*/
    0, 2, 3, 6, 7, 1, 4, 5, /* 205*/
    1, 2, 3, 6, 7, 0, 4, 5, /* 206*/
    0, 1, 2, 3, 6, 7, 4, 5, /* 207*/
    4, 6, 7, 0, 1, 2, 3, 5, /* 208*/
    0, 4, 6, 7, 1, 2, 3, 5, /* 209*/
    1, 4, 6, 7, 0, 2, 3, 5, /* 210*/
    0, 1, 4, 6, 7, 2, 3, 5, /* 211*/
    2, 4, 6, 7, 0, 1, 3, 5, /* 212*/
    0, 2, 4, 6, 7, 1, 3, 5, /* 213*/
    1, 2, 4, 6, 7, 0, 3, 5, /* 214*/
    0, 1, 2, 4, 6, 7, 3, 5, /* 215*/
    3, 4, 6, 7, 0, 1, 2, 5, /* 216*/
    0, 3, 4, 6, 7, 1, 2, 5, /* 217*/
    1, 3, 4, 6, 7, 0, 2, 5, /* 218*/
    0, 1, 3, 4, 6, 7, 2, 5, /* 219*/
    2, 3, 4, 6, 7, 0, 1, 5, /* 220*/
    0, 2, 3, 4, 6, 7, 1, 5, /* 221*/
    1, 2, 3, 4, 6, 7, 0, 5, /* 222*/
    0, 1, 2, 3, 4, 6, 7, 5, /* 223*/
    5, 6, 7, 0, 1, 2, 3, 4, /* 224*/
    0, 5, 6, 7, 1, 2, 3, 4, /* 225*/
    1, 5, 6, 7, 0, 2, 3, 4, /* 226*/
    0, 1, 5, 6, 7, 2, 3, 4, /* 227*/
    2, 5, 6, 7, 0, 1, 3, 4, /* 228*/
    0, 2, 5, 6, 7, 1, 3, 4, /* 229*/
    1, 2, 5, 6, 7, 0, 3, 4, /* 230*/
    0, 1, 2, 5, 6, 7, 3, 4, /* 231*/
    3, 5, 6, 7, 0, 1, 2, 4, /* 232*/
    0, 3, 5, 6, 7, 1, 2, 4, /* 233*/
    1, 3, 5, 6, 7, 0, 2, 4, /* 234*/
    0, 1, 3, 5, 6, 7, 2, 4, /* 235*/
    2, 3, 5, 6, 7, 0, 1, 4, /* 236*/
    0, 2, 3, 5, 6, 7, 1, 4, /* 237*/
    1, 2, 3, 5, 6, 7, 0, 4, /* 238*/
    0, 1, 2, 3, 5, 6, 7, 4, /* 239*/
    4, 5, 6, 7, 0, 1, 2, 3, /* 240*/
    0, 4, 5, 6, 7, 1, 2, 3, /* 241*/
    1, 4, 5, 6, 7, 0, 2, 3, /* 242*/
    0, 1, 4, 5, 6, 7, 2, 3, /* 243*/
    2, 4, 5, 6, 7, 0, 1, 3, /* 244*/
    0, 2, 4, 5, 6, 7, 1, 3, /* 245*/
    1, 2, 4, 5, 6, 7, 0, 3, /* 246*/
    0, 1, 2, 4, 5, 6, 7, 3, /* 247*/
    3, 4, 5, 6, 7, 0, 1, 2, /* 248*/
    0, 3, 4, 5, 6, 7, 1, 2, /* 249*/
    1, 3, 4, 5, 6, 7, 0, 2, /* 250*/
    0, 1, 3, 4, 5, 6, 7, 2, /* 251*/
    2, 3, 4, 5, 6, 7, 0, 1, /* 252*/
    0, 2, 3, 4, 5, 6, 7, 1, /* 253*/
    1, 2, 3, 4, 5, 6, 7, 0, /* 254*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 255*/
};

static inline __m256i _mm256_cmplt_epi32(__m256i a, __m256i b) {
  return _mm256_cmpgt_epi32(b, a);
}


static void avx_natenodutch_partition_epi32(int32_t *array, const int32_t pivot,
                                            int &left, int &right) {
  const __m256i P = _mm256_set1_epi32(pivot);
  const int valuespervector = sizeof(__m256i) / sizeof(int32_t);
  if (right - left + 1 <
      4 * valuespervector) { // not enough space for nate's algo to make sense, falling back
    scalar_partition(array, pivot, left, right);
    return;
  }
  int readleft = left + valuespervector;
  int readright = right - valuespervector;
  int32_t buffer[2 * valuespervector]; // tmp buffer
  memcpy(buffer, array + left, valuespervector * sizeof(int32_t));
  memcpy(buffer + valuespervector, array + right - valuespervector + 1,
         valuespervector * sizeof(int32_t));
  while (readright - readleft >= valuespervector) {
    __m256i *nextVectorPtr;
    if ((readleft - left) > (right - readright)) {
      nextVectorPtr = (__m256i *)(array + readright - valuespervector + 1);
      readright -= valuespervector;
    } else {
      nextVectorPtr = (__m256i *)(array + readleft);
      readleft += valuespervector;
    }
    __m256i currentVector = _mm256_loadu_si256(nextVectorPtr);
    /******
    * we need two comparisons if we are to avoid the Dutch national flag
    * problem,
    * Reference: https://en.wikipedia.org/wiki/Dutch_national_flag_problem
    */
    int greaterthanpivot =
        _mm256_movemask_ps((__m256)_mm256_cmpgt_epi32(currentVector, P));
    int lesserthanpivot =
        _mm256_movemask_ps((__m256)_mm256_cmplt_epi32(currentVector, P));
    __m256i greaterthanpivot_permvector = _mm256_load_si256(
        (__m256i *)(reverseshufflemask + 8 * greaterthanpivot));
    __m256i lesserthanpivot_permvector =
        _mm256_load_si256((__m256i *)(shufflemask + 8 * lesserthanpivot));

    int count_greaterthanpivot = _mm_popcnt_u32(greaterthanpivot);
    int count_lesserthanpivot = _mm_popcnt_u32(lesserthanpivot);
    __m256i greaterthanpivot_vector =
        _mm256_permutevar8x32_epi32(currentVector, greaterthanpivot_permvector);
    __m256i lesserthanpivot_vector =
        _mm256_permutevar8x32_epi32(currentVector, lesserthanpivot_permvector);
    _mm256_storeu_si256((__m256i *)(array + left), lesserthanpivot_vector);

    _mm256_storeu_si256((__m256i *)(array + right - valuespervector + 1),
                        greaterthanpivot_vector);

    left += count_lesserthanpivot;
    right -= count_greaterthanpivot;
  }

  int howmanyleft = readleft - left;
  if (howmanyleft > 2 * valuespervector) {
    howmanyleft = 2 * valuespervector;
    for (int k = howmanyleft; k < readleft - left; ++k)
      array[left + k] = pivot;
  }
  memcpy(array + left, buffer, howmanyleft * sizeof(int32_t));

  int howmanyright = right - readright;
  if (howmanyright + howmanyleft > 2 * valuespervector) {
    howmanyright = 2 * valuespervector - howmanyleft;
    for (int k = howmanyright; k < right - readright; ++k)
      array[readright + 1 + k] = pivot;
  }

  memcpy(array + readright + 1, buffer + howmanyleft,
         howmanyright * sizeof(int32_t));
  scalar_partition(array, pivot, left, right);
}


// it is really signed int sorting, but the API seems to expect uint32_t...
void avx_natenodutch_quicksort(uint32_t *array, int left, int right) {

  int i = left;
  int j = right;
  const int32_t pivot = array[(i + j) / 2];

  avx_natenodutch_partition_epi32((int32_t *) array, pivot, i, j);

  if (left < j) {
    avx_natenodutch_quicksort(array, left, j);
  }

  if (i < right) {
    avx_natenodutch_quicksort(array, i, right);
  }
}
