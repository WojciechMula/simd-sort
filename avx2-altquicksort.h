#pragma once

#include <x86intrin.h>
#include <cstdint>

/**
* This is an alternative approach to SIMD quicksort, implemented by D. Lemire.
* It is not meant to be fast as such, but it can serve as a useful reference.
*/

// can be replaced with VCOMPRESS on AVX-512
static uint32_t reverseshufflemask[256 * 8] __attribute__((aligned(0x100))) = {
    0, 1, 2, 3, 4, 5, 6, 7, /* 0*/
    1, 2, 3, 4, 5, 6, 7, 0, /* 1*/
    0, 2, 3, 4, 5, 6, 7, 1, /* 2*/
    2, 3, 4, 5, 6, 7, 0, 1, /* 3*/
    0, 1, 3, 4, 5, 6, 7, 2, /* 4*/
    1, 3, 4, 5, 6, 7, 0, 2, /* 5*/
    0, 3, 4, 5, 6, 7, 1, 2, /* 6*/
    3, 4, 5, 6, 7, 0, 1, 2, /* 7*/
    0, 1, 2, 4, 5, 6, 7, 3, /* 8*/
    1, 2, 4, 5, 6, 7, 0, 3, /* 9*/
    0, 2, 4, 5, 6, 7, 1, 3, /* 10*/
    2, 4, 5, 6, 7, 0, 1, 3, /* 11*/
    0, 1, 4, 5, 6, 7, 2, 3, /* 12*/
    1, 4, 5, 6, 7, 0, 2, 3, /* 13*/
    0, 4, 5, 6, 7, 1, 2, 3, /* 14*/
    4, 5, 6, 7, 0, 1, 2, 3, /* 15*/
    0, 1, 2, 3, 5, 6, 7, 4, /* 16*/
    1, 2, 3, 5, 6, 7, 0, 4, /* 17*/
    0, 2, 3, 5, 6, 7, 1, 4, /* 18*/
    2, 3, 5, 6, 7, 0, 1, 4, /* 19*/
    0, 1, 3, 5, 6, 7, 2, 4, /* 20*/
    1, 3, 5, 6, 7, 0, 2, 4, /* 21*/
    0, 3, 5, 6, 7, 1, 2, 4, /* 22*/
    3, 5, 6, 7, 0, 1, 2, 4, /* 23*/
    0, 1, 2, 5, 6, 7, 3, 4, /* 24*/
    1, 2, 5, 6, 7, 0, 3, 4, /* 25*/
    0, 2, 5, 6, 7, 1, 3, 4, /* 26*/
    2, 5, 6, 7, 0, 1, 3, 4, /* 27*/
    0, 1, 5, 6, 7, 2, 3, 4, /* 28*/
    1, 5, 6, 7, 0, 2, 3, 4, /* 29*/
    0, 5, 6, 7, 1, 2, 3, 4, /* 30*/
    5, 6, 7, 0, 1, 2, 3, 4, /* 31*/
    0, 1, 2, 3, 4, 6, 7, 5, /* 32*/
    1, 2, 3, 4, 6, 7, 0, 5, /* 33*/
    0, 2, 3, 4, 6, 7, 1, 5, /* 34*/
    2, 3, 4, 6, 7, 0, 1, 5, /* 35*/
    0, 1, 3, 4, 6, 7, 2, 5, /* 36*/
    1, 3, 4, 6, 7, 0, 2, 5, /* 37*/
    0, 3, 4, 6, 7, 1, 2, 5, /* 38*/
    3, 4, 6, 7, 0, 1, 2, 5, /* 39*/
    0, 1, 2, 4, 6, 7, 3, 5, /* 40*/
    1, 2, 4, 6, 7, 0, 3, 5, /* 41*/
    0, 2, 4, 6, 7, 1, 3, 5, /* 42*/
    2, 4, 6, 7, 0, 1, 3, 5, /* 43*/
    0, 1, 4, 6, 7, 2, 3, 5, /* 44*/
    1, 4, 6, 7, 0, 2, 3, 5, /* 45*/
    0, 4, 6, 7, 1, 2, 3, 5, /* 46*/
    4, 6, 7, 0, 1, 2, 3, 5, /* 47*/
    0, 1, 2, 3, 6, 7, 4, 5, /* 48*/
    1, 2, 3, 6, 7, 0, 4, 5, /* 49*/
    0, 2, 3, 6, 7, 1, 4, 5, /* 50*/
    2, 3, 6, 7, 0, 1, 4, 5, /* 51*/
    0, 1, 3, 6, 7, 2, 4, 5, /* 52*/
    1, 3, 6, 7, 0, 2, 4, 5, /* 53*/
    0, 3, 6, 7, 1, 2, 4, 5, /* 54*/
    3, 6, 7, 0, 1, 2, 4, 5, /* 55*/
    0, 1, 2, 6, 7, 3, 4, 5, /* 56*/
    1, 2, 6, 7, 0, 3, 4, 5, /* 57*/
    0, 2, 6, 7, 1, 3, 4, 5, /* 58*/
    2, 6, 7, 0, 1, 3, 4, 5, /* 59*/
    0, 1, 6, 7, 2, 3, 4, 5, /* 60*/
    1, 6, 7, 0, 2, 3, 4, 5, /* 61*/
    0, 6, 7, 1, 2, 3, 4, 5, /* 62*/
    6, 7, 0, 1, 2, 3, 4, 5, /* 63*/
    0, 1, 2, 3, 4, 5, 7, 6, /* 64*/
    1, 2, 3, 4, 5, 7, 0, 6, /* 65*/
    0, 2, 3, 4, 5, 7, 1, 6, /* 66*/
    2, 3, 4, 5, 7, 0, 1, 6, /* 67*/
    0, 1, 3, 4, 5, 7, 2, 6, /* 68*/
    1, 3, 4, 5, 7, 0, 2, 6, /* 69*/
    0, 3, 4, 5, 7, 1, 2, 6, /* 70*/
    3, 4, 5, 7, 0, 1, 2, 6, /* 71*/
    0, 1, 2, 4, 5, 7, 3, 6, /* 72*/
    1, 2, 4, 5, 7, 0, 3, 6, /* 73*/
    0, 2, 4, 5, 7, 1, 3, 6, /* 74*/
    2, 4, 5, 7, 0, 1, 3, 6, /* 75*/
    0, 1, 4, 5, 7, 2, 3, 6, /* 76*/
    1, 4, 5, 7, 0, 2, 3, 6, /* 77*/
    0, 4, 5, 7, 1, 2, 3, 6, /* 78*/
    4, 5, 7, 0, 1, 2, 3, 6, /* 79*/
    0, 1, 2, 3, 5, 7, 4, 6, /* 80*/
    1, 2, 3, 5, 7, 0, 4, 6, /* 81*/
    0, 2, 3, 5, 7, 1, 4, 6, /* 82*/
    2, 3, 5, 7, 0, 1, 4, 6, /* 83*/
    0, 1, 3, 5, 7, 2, 4, 6, /* 84*/
    1, 3, 5, 7, 0, 2, 4, 6, /* 85*/
    0, 3, 5, 7, 1, 2, 4, 6, /* 86*/
    3, 5, 7, 0, 1, 2, 4, 6, /* 87*/
    0, 1, 2, 5, 7, 3, 4, 6, /* 88*/
    1, 2, 5, 7, 0, 3, 4, 6, /* 89*/
    0, 2, 5, 7, 1, 3, 4, 6, /* 90*/
    2, 5, 7, 0, 1, 3, 4, 6, /* 91*/
    0, 1, 5, 7, 2, 3, 4, 6, /* 92*/
    1, 5, 7, 0, 2, 3, 4, 6, /* 93*/
    0, 5, 7, 1, 2, 3, 4, 6, /* 94*/
    5, 7, 0, 1, 2, 3, 4, 6, /* 95*/
    0, 1, 2, 3, 4, 7, 5, 6, /* 96*/
    1, 2, 3, 4, 7, 0, 5, 6, /* 97*/
    0, 2, 3, 4, 7, 1, 5, 6, /* 98*/
    2, 3, 4, 7, 0, 1, 5, 6, /* 99*/
    0, 1, 3, 4, 7, 2, 5, 6, /* 100*/
    1, 3, 4, 7, 0, 2, 5, 6, /* 101*/
    0, 3, 4, 7, 1, 2, 5, 6, /* 102*/
    3, 4, 7, 0, 1, 2, 5, 6, /* 103*/
    0, 1, 2, 4, 7, 3, 5, 6, /* 104*/
    1, 2, 4, 7, 0, 3, 5, 6, /* 105*/
    0, 2, 4, 7, 1, 3, 5, 6, /* 106*/
    2, 4, 7, 0, 1, 3, 5, 6, /* 107*/
    0, 1, 4, 7, 2, 3, 5, 6, /* 108*/
    1, 4, 7, 0, 2, 3, 5, 6, /* 109*/
    0, 4, 7, 1, 2, 3, 5, 6, /* 110*/
    4, 7, 0, 1, 2, 3, 5, 6, /* 111*/
    0, 1, 2, 3, 7, 4, 5, 6, /* 112*/
    1, 2, 3, 7, 0, 4, 5, 6, /* 113*/
    0, 2, 3, 7, 1, 4, 5, 6, /* 114*/
    2, 3, 7, 0, 1, 4, 5, 6, /* 115*/
    0, 1, 3, 7, 2, 4, 5, 6, /* 116*/
    1, 3, 7, 0, 2, 4, 5, 6, /* 117*/
    0, 3, 7, 1, 2, 4, 5, 6, /* 118*/
    3, 7, 0, 1, 2, 4, 5, 6, /* 119*/
    0, 1, 2, 7, 3, 4, 5, 6, /* 120*/
    1, 2, 7, 0, 3, 4, 5, 6, /* 121*/
    0, 2, 7, 1, 3, 4, 5, 6, /* 122*/
    2, 7, 0, 1, 3, 4, 5, 6, /* 123*/
    0, 1, 7, 2, 3, 4, 5, 6, /* 124*/
    1, 7, 0, 2, 3, 4, 5, 6, /* 125*/
    0, 7, 1, 2, 3, 4, 5, 6, /* 126*/
    7, 0, 1, 2, 3, 4, 5, 6, /* 127*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 128*/
    1, 2, 3, 4, 5, 6, 0, 7, /* 129*/
    0, 2, 3, 4, 5, 6, 1, 7, /* 130*/
    2, 3, 4, 5, 6, 0, 1, 7, /* 131*/
    0, 1, 3, 4, 5, 6, 2, 7, /* 132*/
    1, 3, 4, 5, 6, 0, 2, 7, /* 133*/
    0, 3, 4, 5, 6, 1, 2, 7, /* 134*/
    3, 4, 5, 6, 0, 1, 2, 7, /* 135*/
    0, 1, 2, 4, 5, 6, 3, 7, /* 136*/
    1, 2, 4, 5, 6, 0, 3, 7, /* 137*/
    0, 2, 4, 5, 6, 1, 3, 7, /* 138*/
    2, 4, 5, 6, 0, 1, 3, 7, /* 139*/
    0, 1, 4, 5, 6, 2, 3, 7, /* 140*/
    1, 4, 5, 6, 0, 2, 3, 7, /* 141*/
    0, 4, 5, 6, 1, 2, 3, 7, /* 142*/
    4, 5, 6, 0, 1, 2, 3, 7, /* 143*/
    0, 1, 2, 3, 5, 6, 4, 7, /* 144*/
    1, 2, 3, 5, 6, 0, 4, 7, /* 145*/
    0, 2, 3, 5, 6, 1, 4, 7, /* 146*/
    2, 3, 5, 6, 0, 1, 4, 7, /* 147*/
    0, 1, 3, 5, 6, 2, 4, 7, /* 148*/
    1, 3, 5, 6, 0, 2, 4, 7, /* 149*/
    0, 3, 5, 6, 1, 2, 4, 7, /* 150*/
    3, 5, 6, 0, 1, 2, 4, 7, /* 151*/
    0, 1, 2, 5, 6, 3, 4, 7, /* 152*/
    1, 2, 5, 6, 0, 3, 4, 7, /* 153*/
    0, 2, 5, 6, 1, 3, 4, 7, /* 154*/
    2, 5, 6, 0, 1, 3, 4, 7, /* 155*/
    0, 1, 5, 6, 2, 3, 4, 7, /* 156*/
    1, 5, 6, 0, 2, 3, 4, 7, /* 157*/
    0, 5, 6, 1, 2, 3, 4, 7, /* 158*/
    5, 6, 0, 1, 2, 3, 4, 7, /* 159*/
    0, 1, 2, 3, 4, 6, 5, 7, /* 160*/
    1, 2, 3, 4, 6, 0, 5, 7, /* 161*/
    0, 2, 3, 4, 6, 1, 5, 7, /* 162*/
    2, 3, 4, 6, 0, 1, 5, 7, /* 163*/
    0, 1, 3, 4, 6, 2, 5, 7, /* 164*/
    1, 3, 4, 6, 0, 2, 5, 7, /* 165*/
    0, 3, 4, 6, 1, 2, 5, 7, /* 166*/
    3, 4, 6, 0, 1, 2, 5, 7, /* 167*/
    0, 1, 2, 4, 6, 3, 5, 7, /* 168*/
    1, 2, 4, 6, 0, 3, 5, 7, /* 169*/
    0, 2, 4, 6, 1, 3, 5, 7, /* 170*/
    2, 4, 6, 0, 1, 3, 5, 7, /* 171*/
    0, 1, 4, 6, 2, 3, 5, 7, /* 172*/
    1, 4, 6, 0, 2, 3, 5, 7, /* 173*/
    0, 4, 6, 1, 2, 3, 5, 7, /* 174*/
    4, 6, 0, 1, 2, 3, 5, 7, /* 175*/
    0, 1, 2, 3, 6, 4, 5, 7, /* 176*/
    1, 2, 3, 6, 0, 4, 5, 7, /* 177*/
    0, 2, 3, 6, 1, 4, 5, 7, /* 178*/
    2, 3, 6, 0, 1, 4, 5, 7, /* 179*/
    0, 1, 3, 6, 2, 4, 5, 7, /* 180*/
    1, 3, 6, 0, 2, 4, 5, 7, /* 181*/
    0, 3, 6, 1, 2, 4, 5, 7, /* 182*/
    3, 6, 0, 1, 2, 4, 5, 7, /* 183*/
    0, 1, 2, 6, 3, 4, 5, 7, /* 184*/
    1, 2, 6, 0, 3, 4, 5, 7, /* 185*/
    0, 2, 6, 1, 3, 4, 5, 7, /* 186*/
    2, 6, 0, 1, 3, 4, 5, 7, /* 187*/
    0, 1, 6, 2, 3, 4, 5, 7, /* 188*/
    1, 6, 0, 2, 3, 4, 5, 7, /* 189*/
    0, 6, 1, 2, 3, 4, 5, 7, /* 190*/
    6, 0, 1, 2, 3, 4, 5, 7, /* 191*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 192*/
    1, 2, 3, 4, 5, 0, 6, 7, /* 193*/
    0, 2, 3, 4, 5, 1, 6, 7, /* 194*/
    2, 3, 4, 5, 0, 1, 6, 7, /* 195*/
    0, 1, 3, 4, 5, 2, 6, 7, /* 196*/
    1, 3, 4, 5, 0, 2, 6, 7, /* 197*/
    0, 3, 4, 5, 1, 2, 6, 7, /* 198*/
    3, 4, 5, 0, 1, 2, 6, 7, /* 199*/
    0, 1, 2, 4, 5, 3, 6, 7, /* 200*/
    1, 2, 4, 5, 0, 3, 6, 7, /* 201*/
    0, 2, 4, 5, 1, 3, 6, 7, /* 202*/
    2, 4, 5, 0, 1, 3, 6, 7, /* 203*/
    0, 1, 4, 5, 2, 3, 6, 7, /* 204*/
    1, 4, 5, 0, 2, 3, 6, 7, /* 205*/
    0, 4, 5, 1, 2, 3, 6, 7, /* 206*/
    4, 5, 0, 1, 2, 3, 6, 7, /* 207*/
    0, 1, 2, 3, 5, 4, 6, 7, /* 208*/
    1, 2, 3, 5, 0, 4, 6, 7, /* 209*/
    0, 2, 3, 5, 1, 4, 6, 7, /* 210*/
    2, 3, 5, 0, 1, 4, 6, 7, /* 211*/
    0, 1, 3, 5, 2, 4, 6, 7, /* 212*/
    1, 3, 5, 0, 2, 4, 6, 7, /* 213*/
    0, 3, 5, 1, 2, 4, 6, 7, /* 214*/
    3, 5, 0, 1, 2, 4, 6, 7, /* 215*/
    0, 1, 2, 5, 3, 4, 6, 7, /* 216*/
    1, 2, 5, 0, 3, 4, 6, 7, /* 217*/
    0, 2, 5, 1, 3, 4, 6, 7, /* 218*/
    2, 5, 0, 1, 3, 4, 6, 7, /* 219*/
    0, 1, 5, 2, 3, 4, 6, 7, /* 220*/
    1, 5, 0, 2, 3, 4, 6, 7, /* 221*/
    0, 5, 1, 2, 3, 4, 6, 7, /* 222*/
    5, 0, 1, 2, 3, 4, 6, 7, /* 223*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 224*/
    1, 2, 3, 4, 0, 5, 6, 7, /* 225*/
    0, 2, 3, 4, 1, 5, 6, 7, /* 226*/
    2, 3, 4, 0, 1, 5, 6, 7, /* 227*/
    0, 1, 3, 4, 2, 5, 6, 7, /* 228*/
    1, 3, 4, 0, 2, 5, 6, 7, /* 229*/
    0, 3, 4, 1, 2, 5, 6, 7, /* 230*/
    3, 4, 0, 1, 2, 5, 6, 7, /* 231*/
    0, 1, 2, 4, 3, 5, 6, 7, /* 232*/
    1, 2, 4, 0, 3, 5, 6, 7, /* 233*/
    0, 2, 4, 1, 3, 5, 6, 7, /* 234*/
    2, 4, 0, 1, 3, 5, 6, 7, /* 235*/
    0, 1, 4, 2, 3, 5, 6, 7, /* 236*/
    1, 4, 0, 2, 3, 5, 6, 7, /* 237*/
    0, 4, 1, 2, 3, 5, 6, 7, /* 238*/
    4, 0, 1, 2, 3, 5, 6, 7, /* 239*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 240*/
    1, 2, 3, 0, 4, 5, 6, 7, /* 241*/
    0, 2, 3, 1, 4, 5, 6, 7, /* 242*/
    2, 3, 0, 1, 4, 5, 6, 7, /* 243*/
    0, 1, 3, 2, 4, 5, 6, 7, /* 244*/
    1, 3, 0, 2, 4, 5, 6, 7, /* 245*/
    0, 3, 1, 2, 4, 5, 6, 7, /* 246*/
    3, 0, 1, 2, 4, 5, 6, 7, /* 247*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 248*/
    1, 2, 0, 3, 4, 5, 6, 7, /* 249*/
    0, 2, 1, 3, 4, 5, 6, 7, /* 250*/
    2, 0, 1, 3, 4, 5, 6, 7, /* 251*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 252*/
    1, 0, 2, 3, 4, 5, 6, 7, /* 253*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 254*/
    0, 1, 2, 3, 4, 5, 6, 7, /* 255*/
};

static uint32_t avx_pivot_on_last_value(int32_t *array, size_t length) {
  /* we run through the data. Anything in [0,boundary) is smaller or equal
  * than the pivot, and the value at boundary - 1 is going to be equal to the
  * pivot at the end,
  * anything in (boundary, i) is greater than the pivot
  * stuff in [i,...) is grey
  * the function returns the location of the boundary.
  */
  if (length <= 1)
    return 1;
  { // we exchange the last value for the middle value for a better pivot
    int32_t ival = array[length / 2];
    int32_t bval = array[length - 1];
    array[length / 2] = bval;
    array[length - 1] = ival;
  }
#if WITH_RUNTIME_STATS
  statistics.partition_calls += 1;
  statistics.items_processed += length;
#endif
  uint32_t boundary = 0;
  uint32_t i = 0;
  int32_t pivot = array[length - 1]; // we always pick the pivot at the end
  const __m256i P = _mm256_set1_epi32(pivot);
  while ( i + 8 + 1 <= length) {
      __m256i allgrey = _mm256_lddqu_si256((__m256i *)(array + i));
      int pvbyte = _mm256_movemask_ps((__m256)_mm256_cmpgt_epi32(allgrey, P));
      if(pvbyte == 0) { // might be frequent
        i += 8; //nothing to do
        boundary = i;
      } else if (pvbyte == 0xFF) { // called once
        boundary = i;
        i += 8;
        break; // exit
      } else {

        // hot path
        switch (pvbyte) {
            // for pvbyte = 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff
            //              there is no change in order, just advance boundary
            //              Note: case 0x00 & 0xff are already handled
            case 0x80: i += 8 - 1; break;
            case 0xc0: i += 8 - 2; break;
            case 0xe0: i += 8 - 3; break;
            case 0xf0: i += 8 - 4; break;
            case 0xf8: i += 8 - 5; break;
            case 0xfc: i += 8 - 6; break;
            case 0xfe: i += 8 - 7; break;

            // for pvbyte = 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 07f
            //              higher part is swap with lower, no extra permutation is done
            // 
            // Note: pairs 0x01 & 0x7f, 0x03 & 0x3f and 0x7 & 0x1f share swap code.
            case 0x01: {

                    const uint32_t w0 = array[i + 0];
                    const uint32_t w7 = array[i + 7];
                    array[i + 0] = w7;
                    array[i + 7] = w0;

                    i += 8 - 1;
                }
                break;

            case 0x7f: {

                    const uint32_t w0 = array[i + 0];
                    const uint32_t w7 = array[i + 7];
                    array[i + 0] = w7;
                    array[i + 7] = w0;

                    i += 8 - 7;
                }
                break;

            case 0x03: {

                    const uint64_t w01 = *reinterpret_cast<uint64_t*>(array + i + 0);
                    const uint64_t w67 = *reinterpret_cast<uint64_t*>(array + i + 6);

                    *reinterpret_cast<uint64_t*>(array + i + 0) = w67;
                    *reinterpret_cast<uint64_t*>(array + i + 6) = w01;

                    i += 8 - 2;
                }
                break;

            case 0x3f: {

                    const uint64_t w01 = *reinterpret_cast<uint64_t*>(array + i + 0);
                    const uint64_t w67 = *reinterpret_cast<uint64_t*>(array + i + 6);

                    *reinterpret_cast<uint64_t*>(array + i + 0) = w67;
                    *reinterpret_cast<uint64_t*>(array + i + 6) = w01;

                    i += 8 - 6;
                }
                break;

            case 0x07: {

                    const uint64_t w01 = *reinterpret_cast<uint64_t*>(array + i + 0);
                    const uint32_t w2  = array[i + 2];
                    const uint64_t w67 = *reinterpret_cast<uint64_t*>(array + i + 6);
                    const uint32_t w5  = array[i + 5];

                    *reinterpret_cast<uint64_t*>(array + i + 0) = w67;
                    array[i + 2] = w5;
                    *reinterpret_cast<uint64_t*>(array + i + 6) = w01;
                    array[i + 5] = w2;

                    i += 8 - 3;
                }
                break;

            case 0x1f: {

                    const uint64_t w01 = *reinterpret_cast<uint64_t*>(array + i + 0);
                    const uint32_t w2  = array[i + 2];
                    const uint64_t w67 = *reinterpret_cast<uint64_t*>(array + i + 6);
                    const uint32_t w5  = array[i + 5];

                    *reinterpret_cast<uint64_t*>(array + i + 0) = w67;
                    array[i + 2] = w5;
                    *reinterpret_cast<uint64_t*>(array + i + 6) = w01;
                    array[i + 5] = w2;

                    i += 8 - 5;
                }
                break;

            case 0x0f: {
                    // qword order: 2, 3, 0, 1 (0b0001_1011)
                    const __m256i swap = _mm256_permute4x64_epi64(allgrey, 0x1b);
                    _mm256_storeu_si256((__m256i *)(array + i), swap);
                    i += 8 - 4;
                }
                break;

            default: {
                printf("@@@ %02x\n", pvbyte);
              __m256i shufm =
                  _mm256_load_si256((__m256i *)(reverseshufflemask + 8 * pvbyte));
              uint32_t cnt =
                  8 - _mm_popcnt_u32(pvbyte); // might be faster with table look-up?
              __m256i blackthenwhite = _mm256_permutevar8x32_epi32(allgrey, shufm);
              _mm256_storeu_si256((__m256i *)(array + i), blackthenwhite);
              i += cnt;
            }
        } // switch

        boundary = i; // this doesn't need updating each and every time
    }
  }
  for (; i + 8 + 1 <= length ;) {
      __m256i allgrey =
          _mm256_lddqu_si256((__m256i *)(array + i)); // this is all grey
      int pvbyte = _mm256_movemask_ps((__m256)_mm256_cmpgt_epi32(allgrey, P));
      if (pvbyte == 0xFF) { // called once
        // nothing to do
      } else {

      __m256i shufm =
          _mm256_load_si256((__m256i *)(reverseshufflemask + 8 * pvbyte));
      uint32_t cnt =
          8 - _mm_popcnt_u32(pvbyte); // might be faster with table look-up?
      __m256i allwhite = _mm256_lddqu_si256(
          (__m256i *)(array + boundary)); // this is all white
      // we shuffle allgrey so that the first part is black and the second part
      // is white
      __m256i blackthenwhite = _mm256_permutevar8x32_epi32(allgrey, shufm);
      _mm256_storeu_si256((__m256i *)(array + boundary), blackthenwhite);
      _mm256_storeu_si256((__m256i *)(array + i), allwhite);
      boundary += cnt; // might be faster with table look-up?
    }
      i += 8;
  }
  while (i + 1 < length) {
    int32_t ival = array[i];
    if (ival <= pivot) {
      int32_t bval = array[boundary];
      array[i] = bval;
      array[boundary] = ival;
      boundary++;
    }
    i++;
  }
  int32_t ival = array[i];
  int32_t bval = array[boundary];
  array[length - 1] = bval;
  array[boundary] = ival;
  boundary++;
  return boundary;
}

// for fallback
void scalar_partition(int32_t* array, const int32_t pivot, int& left, int& right) {

    while (left <= right) {
        while (array[left] < pivot) {
            left += 1;
        }
        while (array[right] > pivot) {
            right -= 1;
        }
        if (left <= right) {
            const uint32_t t = array[left];
            array[left]      = array[right];
            array[right]     = t;
            left  += 1;
            right -= 1;
        }
    }
}

//fallback
void scalar_quicksort(int32_t* array, int left, int right) {
#ifdef WITH_RUNTIME_STATS
    statistics.scalar__partition_calls += 1;
    statistics.scalar__items_processed += right - left + 1;
#endif
    int i = left;
    int j = right;
    const int32_t pivot = array[(i + j)/2];
    scalar_partition(array, pivot, i, j);
    if (left < j) {
        scalar_quicksort(array, left, j);
    }
    if (i < right) {
        scalar_quicksort(array, i, right);
    }
}

void avx2_pivotonlast_sort(int32_t *array, const uint32_t length) {
  uint32_t sep = avx_pivot_on_last_value(array, length);
  if(sep == length) {
    // we have an ineffective pivot. Let us give up.
    if(length > 1) scalar_quicksort(array,0,length - 1);
  } else {
    if (sep > 2) {
      avx2_pivotonlast_sort(array, sep - 1);
    }
    if (sep + 1 < length) {
      avx2_pivotonlast_sort(array + sep, length - sep);
    }
  }
}
void wrapped_avx2_pivotonlast_sort(uint32_t *array, int left, int right) {
  avx2_pivotonlast_sort((int32_t *)array + left, right - left + 1);
}
