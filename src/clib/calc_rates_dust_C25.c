#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "grackle_macros.h"
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "phys_constants.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define tiny 1.0e-20
#define huge 1.0e+20
#define tevk 1.1605e+4

extern int grackle_verbose;

int calc_rates_dust_C25(int iSN, chemistry_data *my_chemistry, chemistry_data_storage *my_rates)
{

  int NTd, Nmom;
  int iTd, imom, itab0, itab;

  my_rates->SN0_XC [iSN] =   1.75488e-01;
  my_rates->SN0_XO [iSN] =   5.69674e-01;
  my_rates->SN0_XMg[iSN] =   3.12340e-02;
  my_rates->SN0_XAl[iSN] =   2.98415e-04;
  my_rates->SN0_XSi[iSN] =   8.33205e-02;
  my_rates->SN0_XS [iSN] =   4.73930e-02;
  my_rates->SN0_XFe[iSN] =   1.98197e-02;

  my_rates->SN0_fC [iSN] =   1.34092e-01;
  my_rates->SN0_fO [iSN] =   5.53726e-01;
  my_rates->SN0_fMg[iSN] =   2.48100e-02;
  my_rates->SN0_fAl[iSN] =   2.98415e-04;
  my_rates->SN0_fSi[iSN] =   3.47760e-02;
  my_rates->SN0_fS [iSN] =   4.72556e-02;
  my_rates->SN0_fFe[iSN] =   1.46955e-02;

  my_rates->SN0_fSiM     [iSN] =   3.83373e-02;
  my_rates->SN0_fFeM     [iSN] =   4.88366e-03;
  my_rates->SN0_fMg2SiO4 [iSN] =   1.68068e-02;
  my_rates->SN0_fMgSiO3  [iSN] =   2.49736e-05;
  my_rates->SN0_fAC      [iSN] =   4.13961e-02;
  my_rates->SN0_fSiO2D   [iSN] =   1.46546e-02;
  my_rates->SN0_fMgO     [iSN] =   1.09289e-03;
  my_rates->SN0_fFeS     [iSN] =   3.77935e-04;
  my_rates->SN0_fAl2O3   [iSN] =   1.65550e-31;

  itab0 = 3 * iSN;
  my_rates->SN0_r0SiM     [itab0 + 0] =   1.72153e-05;
  my_rates->SN0_r0FeM     [itab0 + 0] =   1.96666e-05;
  my_rates->SN0_r0Mg2SiO4 [itab0 + 0] =   2.33213e-06;
  my_rates->SN0_r0MgSiO3  [itab0 + 0] =   1.55439e-06;
  my_rates->SN0_r0AC      [itab0 + 0] =   7.93494e-07;
  my_rates->SN0_r0SiO2D   [itab0 + 0] =   2.56804e-06;
  my_rates->SN0_r0MgO     [itab0 + 0] =   3.58420e-06;
  my_rates->SN0_r0FeS     [itab0 + 0] =   9.61035e-07;
  my_rates->SN0_r0Al2O3   [itab0 + 0] =   1.99526e-08;

  my_rates->SN0_r0SiM     [itab0 + 1] =   6.33208e-10;
  my_rates->SN0_r0FeM     [itab0 + 1] =   5.88305e-10;
  my_rates->SN0_r0Mg2SiO4 [itab0 + 1] =   2.48648e-11;
  my_rates->SN0_r0MgSiO3  [itab0 + 1] =   4.30058e-12;
  my_rates->SN0_r0AC      [itab0 + 1] =   3.53402e-12;
  my_rates->SN0_r0SiO2D   [itab0 + 1] =   4.82971e-11;
  my_rates->SN0_r0MgO     [itab0 + 1] =   3.09713e-11;
  my_rates->SN0_r0FeS     [itab0 + 1] =   2.46507e-12;
  my_rates->SN0_r0Al2O3   [itab0 + 1] =   3.98107e-16;

  my_rates->SN0_r0SiM     [itab0 + 2] =   4.04318e-14;
  my_rates->SN0_r0FeM     [itab0 + 2] =   2.42323e-14;
  my_rates->SN0_r0Mg2SiO4 [itab0 + 2] =   4.29427e-16;
  my_rates->SN0_r0MgSiO3  [itab0 + 2] =   1.92568e-17;
  my_rates->SN0_r0AC      [itab0 + 2] =   1.04050e-16;
  my_rates->SN0_r0SiO2D   [itab0 + 2] =   2.53766e-15;
  my_rates->SN0_r0MgO     [itab0 + 2] =   4.03929e-16;
  my_rates->SN0_r0FeS     [itab0 + 2] =   1.42549e-17;
  my_rates->SN0_r0Al2O3   [itab0 + 2] =   7.94328e-24;

  NTd =            35;
 Nmom =             4;

  double C25_kpSiM[] = 
  {  1.53307e-01,   2.58151e-06,   8.97185e-11,   5.13410e-15,
     1.93187e-01,   3.26103e-06,   1.14053e-10,   6.60852e-15,
     2.43381e-01,   4.11484e-06,   1.44443e-10,   8.42753e-15,
     3.06566e-01,   5.18903e-06,   1.82603e-10,   1.07022e-14,
     3.86268e-01,   6.55217e-06,   2.31853e-10,   1.37374e-14,
     4.86630e-01,   8.26891e-06,   2.93869e-10,   1.75576e-14,
     6.13093e-01,   1.04376e-05,   3.72751e-10,   2.24815e-14,
     7.72441e-01,   1.31779e-05,   4.73205e-10,   2.88483e-14,
     9.72908e-01,   1.66348e-05,   6.00927e-10,   3.70670e-14,
     1.22279e+00,   2.09532e-05,   7.61455e-10,   4.75186e-14,
     1.51967e+00,   2.61010e-05,   9.54694e-10,   6.03376e-14,
     1.83660e+00,   3.16418e-05,   1.16836e-09,   7.52509e-14,
     2.15883e+00,   3.73832e-05,   1.40366e-09,   9.34936e-14,
     2.56188e+00,   4.46730e-05,   1.71564e-09,   1.19386e-13,
     3.24331e+00,   5.69556e-05,   2.23297e-09,   1.61351e-13,
     4.36192e+00,   7.70449e-05,   3.06366e-09,   2.26659e-13,
     5.87089e+00,   1.04185e-04,   4.17746e-09,   3.12654e-13,
     7.58000e+00,   1.35161e-04,   5.44737e-09,   4.09330e-13,
     9.38530e+00,   1.68565e-04,   6.82760e-09,   5.12932e-13,
     1.15371e+01,   2.10285e-04,   8.59830e-09,   6.44799e-13,
     1.47388e+01,   2.75860e-04,   1.14755e-08,   8.58985e-13,
     1.93877e+01,   3.75887e-04,   1.59953e-08,   1.19600e-12,
     2.46008e+01,   5.00971e-04,   2.20360e-08,   1.65571e-12,
     2.90408e+01,   6.45456e-04,   3.01610e-08,   2.31085e-12,
     3.23701e+01,   8.27330e-04,   4.23216e-08,   3.35508e-12,
     3.49305e+01,   1.05238e-03,   5.92405e-08,   4.87268e-12,
     3.68020e+01,   1.28200e-03,   7.79075e-08,   6.59988e-12,
     3.76194e+01,   1.45174e-03,   9.27975e-08,   8.01634e-12,
     3.72617e+01,   1.52446e-03,   1.00177e-07,   8.74632e-12,
     3.76576e+01,   1.55784e-03,   1.02687e-07,   8.96074e-12,
     4.91245e+01,   1.84545e-03,   1.14505e-07,   9.64022e-12,
     1.17100e+02,   3.32870e-03,   1.70057e-07,   1.27411e-11,
     3.93519e+02,   8.31574e-03,   3.27938e-07,   2.07522e-11,
     1.23314e+03,   2.12665e-02,   6.72506e-07,   3.61691e-11,
     3.12736e+03,   4.71942e-02,   1.26825e-06,   5.97225e-11  };

  double C25_kpFeM[] = 
  {  7.05387e-02,   2.70513e-06,   1.33741e-10,   7.96280e-15,
     1.06564e-01,   3.93018e-06,   1.88633e-10,   1.09924e-14,
     1.50619e-01,   5.42584e-06,   2.55756e-10,   1.47090e-14,
     2.05371e-01,   7.28371e-06,   3.39206e-10,   1.93364e-14,
     2.90528e-01,   9.90213e-06,   4.47050e-10,   2.49139e-14,
     3.98319e-01,   1.31460e-05,   5.78557e-10,   3.16454e-14,
     5.44029e-01,   1.73143e-05,   7.40062e-10,   3.96165e-14,
     7.41852e-01,   2.27123e-05,   9.40496e-10,   4.91736e-14,
     1.01111e+00,   2.97427e-05,   1.19128e-09,   6.07485e-14,
     1.37135e+00,   3.87934e-05,   1.50307e-09,   7.47433e-14,
     1.85427e+00,   5.04362e-05,   1.88890e-09,   9.15116e-14,
     2.48538e+00,   6.50566e-05,   2.35529e-09,   1.11141e-13,
     3.28338e+00,   8.28882e-05,   2.90473e-09,   1.33599e-13,
     4.26495e+00,   1.04109e-04,   3.53805e-09,   1.58791e-13,
     5.44209e+00,   1.28847e-04,   4.25649e-09,   1.86722e-13,
     6.82357e+00,   1.57201e-04,   5.06195e-09,   2.17471e-13,
     8.41130e+00,   1.89169e-04,   5.95453e-09,   2.51081e-13,
     1.02033e+01,   2.24714e-04,   6.93482e-09,   2.87663e-13,
     1.22164e+01,   2.64256e-04,   8.01879e-09,   3.27999e-13,
     1.45282e+01,   3.09507e-04,   9.26157e-09,   3.74467e-13,
     1.73262e+01,   3.64447e-04,   1.07851e-08,   4.32120e-13,
     2.09720e+01,   4.36715e-04,   1.28220e-08,   5.10543e-13,
     2.61035e+01,   5.39852e-04,   1.57872e-08,   6.26923e-13,
     3.37725e+01,   6.96147e-04,   2.03646e-08,   8.09666e-13,
     4.55764e+01,   9.38083e-04,   2.75201e-08,   1.09797e-12,
     6.37992e+01,   1.30793e-03,   3.84014e-08,   1.53493e-12,
     9.17796e+01,   1.86084e-03,   5.43135e-08,   2.16276e-12,
     1.34749e+02,   2.67751e-03,   7.69998e-08,   3.03160e-12,
     2.01156e+02,   3.88432e-03,   1.09109e-07,   4.21570e-12,
     3.04328e+02,   5.67395e-03,   1.54550e-07,   5.82179e-12,
     4.64551e+02,   8.32593e-03,   2.18657e-07,   7.98599e-12,
     7.11390e+02,   1.22244e-02,   3.08194e-07,   1.08633e-11,
     1.08442e+03,   1.78487e-02,   4.30797e-07,   1.46043e-11,
     1.62442e+03,   2.56300e-02,   5.92006e-07,   1.92777e-11,
     2.34943e+03,   3.56242e-02,   7.89629e-07,   2.47479e-11  };

  double C25_kpMg2SiO4[] = 
  {  1.05240e-01,   2.45433e-07,   2.61677e-12,   4.51929e-17,
     1.32588e-01,   3.09211e-07,   3.29676e-12,   5.69367e-17,
     1.67016e-01,   3.89504e-07,   4.15283e-12,   7.17213e-17,
     2.10360e-01,   4.90585e-07,   5.23055e-12,   9.03341e-17,
     2.71887e-01,   6.34079e-07,   6.76050e-12,   1.16758e-16,
     3.55694e-01,   8.29533e-07,   8.84446e-12,   1.52750e-16,
     4.84933e-01,   1.13094e-06,   1.20582e-11,   2.08253e-16,
     6.99770e-01,   1.63200e-06,   1.74006e-11,   3.00524e-16,
     1.05860e+00,   2.46891e-06,   2.63246e-11,   4.54655e-16,
     1.62903e+00,   3.79938e-06,   4.05116e-11,   6.99697e-16,
     2.54264e+00,   5.93041e-06,   6.32372e-11,   1.09224e-15,
     3.96499e+00,   9.24857e-06,   9.86269e-11,   1.70359e-15,
     6.10655e+00,   1.42452e-05,   1.51927e-10,   2.62446e-15,
     9.28824e+00,   2.16706e-05,   2.31158e-10,   3.99359e-15,
     1.39278e+01,   3.25027e-05,   3.46790e-10,   5.99240e-15,
     2.05413e+01,   4.79530e-05,   5.11839e-10,   8.84697e-15,
     3.00722e+01,   7.02477e-05,   7.50339e-10,   1.29761e-14,
     4.55290e+01,   1.06478e-04,   1.13881e-09,   1.97131e-14,
     7.48333e+01,   1.75301e-04,   1.87829e-09,   3.25565e-14,
     1.29734e+02,   3.04341e-04,   3.26603e-09,   5.66753e-14,
     2.15039e+02,   5.04949e-04,   5.42460e-09,   9.42056e-14,
     3.20373e+02,   7.52945e-04,   8.09632e-09,   1.40698e-13,
     4.30336e+02,   1.01240e-03,   1.08979e-08,   1.89530e-13,
     5.31620e+02,   1.25190e-03,   1.34895e-08,   2.34771e-13,
     6.00659e+02,   1.41538e-03,   1.52608e-08,   2.65718e-13,
     6.07548e+02,   1.43203e-03,   1.54444e-08,   2.68965e-13,
     5.44184e+02,   1.28282e-03,   1.38365e-08,   2.40975e-13,
     4.34292e+02,   1.02387e-03,   1.10444e-08,   1.92359e-13,
     3.13872e+02,   7.40120e-04,   7.98523e-09,   1.39097e-13,
     2.09392e+02,   4.93970e-04,   5.33184e-09,   9.29066e-14,
     1.31415e+02,   3.10355e-04,   3.35361e-09,   5.84810e-14,
     7.92901e+01,   1.87957e-04,   2.03808e-09,   3.56245e-14,
     4.76038e+01,   1.14252e-04,   1.25235e-09,   2.20472e-14,
     3.00283e+01,   7.42825e-05,   8.34278e-10,   1.49154e-14,
     2.10539e+01,   5.48540e-05,   6.37875e-10,   1.16338e-14  };

  double C25_kpMgSiO3[] = 
  {  2.19890e-02,   3.41795e-08,   9.45655e-14,   4.23439e-19,
     3.90612e-02,   6.07164e-08,   1.67986e-13,   7.52197e-19,
     6.05539e-02,   9.41245e-08,   2.60417e-13,   1.16608e-18,
     8.76116e-02,   1.36183e-07,   3.76781e-13,   1.68713e-18,
     1.43288e-01,   2.22725e-07,   6.16221e-13,   2.75928e-18,
     2.19266e-01,   3.40825e-07,   9.42974e-13,   4.22240e-18,
     3.36256e-01,   5.22673e-07,   1.44610e-12,   6.47526e-18,
     5.14336e-01,   7.99479e-07,   2.21195e-12,   9.90458e-18,
     7.97217e-01,   1.23919e-06,   3.42851e-12,   1.53521e-17,
     1.25414e+00,   1.94943e-06,   5.39358e-12,   2.41515e-17,
     2.03450e+00,   3.16241e-06,   8.74964e-12,   3.91798e-17,
     3.34649e+00,   5.20178e-06,   1.43922e-11,   6.44481e-17,
     5.45897e+00,   8.48547e-06,   2.34778e-11,   1.05137e-16,
     8.82126e+00,   1.37119e-05,   3.79391e-11,   1.69905e-16,
     1.41827e+01,   2.20461e-05,   6.10001e-11,   2.73202e-16,
     2.28425e+01,   3.55077e-05,   9.82519e-11,   4.40095e-16,
     3.71193e+01,   5.77019e-05,   1.59676e-10,   7.15351e-16,
     6.14319e+01,   9.54994e-05,   2.64297e-10,   1.18434e-15,
     1.03856e+02,   1.61458e-04,   4.46893e-10,   2.00314e-15,
     1.75529e+02,   2.72897e-04,   7.55426e-10,   3.38696e-15,
     2.82103e+02,   4.38604e-04,   1.21423e-09,   5.44494e-15,
     4.14591e+02,   6.44612e-04,   1.78465e-09,   8.00387e-15,
     5.60087e+02,   8.70868e-04,   2.41120e-09,   1.08148e-14,
     7.11212e+02,   1.10590e-03,   3.06203e-09,   1.37338e-14,
     8.41053e+02,   1.30782e-03,   3.62100e-09,   1.62382e-14,
     8.95593e+02,   1.39263e-03,   3.85557e-09,   1.72860e-14,
     8.40674e+02,   1.30722e-03,   3.61884e-09,   1.62210e-14,
     6.96909e+02,   1.08365e-03,   2.99976e-09,   1.34436e-14,
     5.18364e+02,   8.06018e-04,   2.23111e-09,   9.99751e-15,
     3.52974e+02,   5.48846e-04,   1.51920e-09,   6.80696e-15,
     2.24287e+02,   3.48752e-04,   9.65345e-10,   4.32526e-15,
     1.35182e+02,   2.10204e-04,   5.81862e-10,   2.60719e-15,
     7.83444e+01,   1.21833e-04,   3.37284e-10,   1.51159e-15,
     4.41840e+01,   6.87277e-05,   1.90332e-10,   8.53442e-16,
     2.46537e+01,   3.84248e-05,   1.06643e-10,   4.79266e-16  };

  double C25_kpAC[] = 
  {  3.27960e-01,   2.60233e-07,   1.15896e-12,   3.41207e-17,
     4.38752e-01,   3.48153e-07,   1.55085e-12,   4.56711e-17,
     5.78230e-01,   4.58837e-07,   2.04421e-12,   6.02121e-17,
     7.53824e-01,   5.98180e-07,   2.66530e-12,   7.85179e-17,
     1.04013e+00,   8.25404e-07,   3.67884e-12,   1.08416e-16,
     1.41735e+00,   1.12479e-06,   5.01451e-12,   1.47828e-16,
     1.95293e+00,   1.54986e-06,   6.91189e-12,   2.03855e-16,
     2.71532e+00,   2.15499e-06,   9.61533e-12,   2.83773e-16,
     3.79678e+00,   3.01350e-06,   1.34554e-11,   3.97470e-16,
     5.29747e+00,   4.20498e-06,   1.87919e-11,   5.55745e-16,
     7.37842e+00,   5.85746e-06,   2.62073e-11,   7.76231e-16,
     1.02169e+01,   8.11222e-06,   3.63532e-11,   1.07888e-15,
     1.40424e+01,   1.11520e-05,   5.00793e-11,   1.48986e-15,
     1.92027e+01,   1.52549e-05,   6.86959e-11,   2.04983e-15,
     2.61627e+01,   2.07929e-05,   9.39920e-11,   2.81528e-15,
     3.55327e+01,   2.82565e-05,   1.28407e-10,   3.86530e-15,
     4.81650e+01,   3.83349e-05,   1.75518e-10,   5.31971e-15,
     6.53244e+01,   5.20561e-05,   2.40904e-10,   7.37173e-15,
     8.87808e+01,   7.08733e-05,   3.33003e-10,   1.03268e-14,
     1.20733e+02,   9.66248e-05,   4.63924e-10,   1.46567e-14,
     1.63674e+02,   1.31491e-04,   6.51645e-10,   2.11366e-14,
     2.20677e+02,   1.78321e-04,   9.26445e-10,   3.11929e-14,
     2.96305e+02,   2.41503e-04,   1.34108e-09,   4.74137e-14,
     3.97405e+02,   3.27522e-04,   1.97072e-09,   7.35006e-14,
     5.32189e+02,   4.43695e-04,   2.88200e-09,   1.12542e-13,
     7.08046e+02,   5.95675e-04,   4.08283e-09,   1.64175e-13,
     9.32526e+02,   7.88039e-04,   5.50980e-09,   2.23600e-13,
     1.21879e+03,   1.02941e-03,   7.08050e-09,   2.84124e-13,
     1.59074e+03,   1.33766e-03,   8.76094e-09,   3.40697e-13,
     2.08510e+03,   1.74270e-03,   1.05986e-08,   3.91613e-13,
     2.75362e+03,   2.28917e-03,   1.27282e-08,   4.38283e-13,
     3.66839e+03,   3.04259e-03,   1.53805e-08,   4.84412e-13,
     4.93379e+03,   4.10065e-03,   1.89004e-08,   5.35144e-13,
     6.70658e+03,   5.60645e-03,   2.37440e-08,   5.95776e-13,
     9.22668e+03,   7.75494e-03,   3.03854e-08,   6.69355e-13  };

  double C25_kpSiO2D[] = 
  {  7.60344e-02,   1.95196e-07,   3.66716e-12,   1.92423e-16,
     9.07191e-02,   2.32906e-07,   4.37632e-12,   2.29682e-16,
     1.09206e-01,   2.80380e-07,   5.26909e-12,   2.76586e-16,
     1.32480e-01,   3.40146e-07,   6.39301e-12,   3.35635e-16,
     1.58906e-01,   4.08019e-07,   7.66999e-12,   4.02759e-16,
     1.91564e-01,   4.91897e-07,   9.24810e-12,   4.85715e-16,
     2.30489e-01,   5.91872e-07,   1.11292e-11,   5.84611e-16,
     2.76795e-01,   7.10808e-07,   1.33674e-11,   7.02310e-16,
     3.33075e-01,   8.55378e-07,   1.60886e-11,   8.45443e-16,
     4.05328e-01,   1.04100e-06,   1.95833e-11,   1.02932e-15,
     5.08167e-01,   1.30521e-06,   2.45600e-11,   1.29133e-15,
     6.72485e-01,   1.72749e-06,   3.25210e-11,   1.71095e-15,
     9.48580e-01,   2.43730e-06,   4.59219e-11,   2.41877e-15,
     1.41796e+00,   3.64482e-06,   6.87751e-11,   3.63027e-15,
     2.19521e+00,   5.64613e-06,   1.06783e-10,   5.65494e-15,
     3.46773e+00,   8.92808e-06,   1.69393e-10,   9.00627e-15,
     5.77034e+00,   1.48887e-05,   2.83967e-10,   1.51653e-14,
     1.17273e+01,   3.03917e-05,   5.84644e-10,   3.13500e-14,
     3.16762e+01,   8.23694e-05,   1.59258e-09,   8.53491e-14,
     8.69213e+01,   2.26080e-04,   4.36458e-09,   2.32841e-13,
     1.92500e+02,   5.00291e-04,   9.62913e-09,   5.11437e-13,
     3.36556e+02,   8.73556e-04,   1.67531e-08,   8.86507e-13,
     5.06180e+02,   1.30807e-03,   2.48422e-08,   1.30609e-12,
     7.20631e+02,   1.84289e-03,   3.42245e-08,   1.77586e-12,
     9.76452e+02,   2.46285e-03,   4.43878e-08,   2.26306e-12,
     1.18428e+03,   2.95079e-03,   5.17620e-08,   2.59653e-12,
     1.23545e+03,   3.05120e-03,   5.24657e-08,   2.59989e-12,
     1.10864e+03,   2.72192e-03,   4.61791e-08,   2.26949e-12,
     8.73605e+02,   2.13674e-03,   3.59368e-08,   1.75670e-12,
     6.20107e+02,   1.51303e-03,   2.53058e-08,   1.23288e-12,
     4.05864e+02,   9.88756e-04,   1.64791e-08,   8.01183e-13,
     2.49677e+02,   6.07668e-04,   1.01055e-08,   4.90701e-13,
     1.46540e+02,   3.56444e-04,   5.92003e-09,   2.87265e-13,
     8.30108e+01,   2.01877e-04,   3.35130e-09,   1.62587e-13,
     4.57957e+01,   1.11407e-04,   1.85045e-09,   8.98118e-14  };

  double C25_kpMgO[] = 
  {  2.25388e-04,   8.07807e-10,   6.97998e-15,   9.10286e-20,
     4.04965e-04,   1.45145e-09,   1.25417e-14,   1.63564e-19,
     6.31040e-04,   2.26174e-09,   1.95435e-14,   2.54881e-19,
     9.15651e-04,   3.28184e-09,   2.83582e-14,   3.69842e-19,
     1.52197e-03,   5.45504e-09,   4.71373e-14,   6.14765e-19,
     2.37408e-03,   8.50921e-09,   7.35292e-14,   9.58978e-19,
     3.77210e-03,   1.35201e-08,   1.16831e-13,   1.52374e-18,
     6.14351e-03,   2.20201e-08,   1.90284e-13,   2.48178e-18,
     1.01908e-02,   3.65275e-08,   3.15657e-13,   4.11707e-18,
     1.68899e-02,   6.05411e-08,   5.23195e-13,   6.82425e-18,
     2.96134e-02,   1.06156e-07,   9.17483e-13,   1.19682e-17,
     6.10648e-02,   2.18932e-07,   1.89256e-12,   2.46925e-17,
     1.43413e-01,   5.14257e-07,   4.44656e-12,   5.80289e-17,
     3.27427e-01,   1.17431e-06,   1.01561e-11,   1.32572e-16,
     6.39567e-01,   2.29431e-06,   1.98486e-11,   2.59169e-16,
     1.05188e+00,   3.77698e-06,   3.27186e-11,   4.27780e-16,
     1.56137e+00,   5.64036e-06,   4.92651e-11,   6.49403e-16,
     2.95878e+00,   1.08379e-05,   9.64135e-11,   1.29373e-15,
     1.33369e+01,   4.85546e-05,   4.28194e-10,   5.69626e-15,
     7.10715e+01,   2.55311e-04,   2.21076e-09,   2.88829e-14,
     2.54519e+02,   9.07495e-04,   7.77715e-09,   1.00557e-13,
     5.99024e+02,   2.12736e-03,   1.81309e-08,   2.33128e-13,
     9.91502e+02,   3.51313e-03,   2.98462e-08,   3.82528e-13,
     1.23853e+03,   4.38206e-03,   3.71546e-08,   4.75243e-13,
     1.24134e+03,   4.38799e-03,   3.71575e-08,   4.74666e-13,
     1.05053e+03,   3.71132e-03,   3.14012e-08,   4.00789e-13,
     7.82169e+02,   2.76204e-03,   2.33559e-08,   2.97932e-13,
     5.28995e+02,   1.86748e-03,   1.57854e-08,   2.01282e-13,
     3.32978e+02,   1.17527e-03,   9.93166e-09,   1.26606e-13,
     1.98632e+02,   7.00988e-04,   5.92262e-09,   7.54858e-14,
     1.13813e+02,   4.01617e-04,   3.39280e-09,   4.32365e-14,
     6.32536e+01,   2.23189e-04,   1.88528e-09,   2.40230e-14,
     3.43507e+01,   1.21200e-04,   1.02371e-09,   1.30436e-14,
     1.83277e+01,   6.46641e-05,   5.46160e-10,   6.95860e-15,
     9.64683e+00,   3.40355e-05,   2.87461e-10,   3.66245e-15  };

  double C25_kpFeS[] = 
  {  5.18089e-02,   4.97944e-08,   1.27767e-13,   7.39409e-19,
     9.98898e-02,   9.60047e-08,   2.46329e-13,   1.42543e-18,
     1.60420e-01,   1.54180e-07,   3.95589e-13,   2.28909e-18,
     2.36623e-01,   2.27418e-07,   5.83496e-13,   3.37637e-18,
     3.67289e-01,   3.53003e-07,   9.05730e-13,   5.24118e-18,
     5.36230e-01,   5.15376e-07,   1.32237e-12,   7.65247e-18,
     7.64209e-01,   7.34494e-07,   1.88465e-12,   1.09071e-17,
     1.04972e+00,   1.00891e-06,   2.58893e-12,   1.49849e-17,
     1.38083e+00,   1.32717e-06,   3.40583e-12,   1.97167e-17,
     1.74377e+00,   1.67607e-06,   4.30174e-12,   2.49099e-17,
     2.10317e+00,   2.02159e-06,   5.18938e-12,   3.00615e-17,
     2.42155e+00,   2.32777e-06,   5.97696e-12,   3.46445e-17,
     2.66826e+00,   2.56519e-06,   6.58920e-12,   3.82277e-17,
     2.81642e+00,   2.70807e-06,   6.96112e-12,   4.04503e-17,
     2.89877e+00,   2.78835e-06,   7.17920e-12,   4.18739e-17,
     3.08081e+00,   2.96627e-06,   7.66777e-12,   4.51307e-17,
     3.69596e+00,   3.56434e-06,   9.27486e-12,   5.54010e-17,
     5.05776e+00,   4.88601e-06,   1.28025e-11,   7.76522e-17,
     7.07898e+00,   6.85096e-06,   1.80991e-11,   1.11855e-16,
     9.20178e+00,   8.93168e-06,   2.39573e-11,   1.53397e-16,
     1.08226e+01,   1.05533e-05,   2.89938e-11,   1.96013e-16,
     1.16924e+01,   1.14675e-05,   3.24020e-11,   2.32497e-16,
     1.19355e+01,   1.17825e-05,   3.42098e-11,   2.58640e-16,
     1.18230e+01,   1.17613e-05,   3.49907e-11,   2.75275e-16,
     1.15929e+01,   1.16487e-05,   3.54453e-11,   2.86649e-16,
     1.13868e+01,   1.16310e-05,   3.63433e-11,   2.99851e-16,
     1.14199e+01,   1.23167e-05,   4.12834e-11,   3.50696e-16,
     1.29371e+01,   1.75436e-05,   7.43145e-11,   6.75730e-16,
     2.05440e+01,   3.98903e-05,   2.16132e-10,   2.04334e-15,
     4.78755e+01,   1.10160e-04,   6.39249e-10,   5.90057e-15,
     1.21252e+02,   2.78758e-04,   1.57840e-09,   1.39121e-14,
     2.69644e+02,   5.88705e-04,   3.18490e-09,   2.68674e-14,
     5.11453e+02,   1.05406e-03,   5.44914e-09,   4.43131e-14,
     8.53209e+02,   1.66585e-03,   8.25940e-09,   6.51454e-14,
     1.29539e+03,   2.40455e-03,   1.14623e-08,   8.80401e-14  };

  double C25_kpAl2O3[] = 
  {  9.93250e-04,   1.98179e-11,   3.95420e-19,   7.88967e-27,
     1.81240e-03,   3.61621e-11,   7.21529e-19,   1.43964e-26,
     2.84365e-03,   5.67382e-11,   1.13208e-18,   2.25879e-26,
     4.14191e-03,   8.26420e-11,   1.64892e-18,   3.29004e-26,
     7.18271e-03,   1.43314e-10,   2.85949e-18,   5.70543e-26,
     1.13364e-02,   2.26190e-10,   4.51309e-18,   9.00479e-26,
     1.77361e-02,   3.53881e-10,   7.06085e-18,   1.40883e-25,
     2.59477e-02,   5.17725e-10,   1.03300e-17,   2.06110e-25,
     3.45425e-02,   6.89214e-10,   1.37516e-17,   2.74381e-25,
     4.22006e-02,   8.42014e-10,   1.68004e-17,   3.35212e-25,
     4.71420e-02,   9.40607e-10,   1.87676e-17,   3.74462e-25,
     4.91934e-02,   9.81537e-10,   1.95842e-17,   3.90757e-25,
     5.05162e-02,   1.00793e-09,   2.01109e-17,   4.01264e-25,
     5.78201e-02,   1.15366e-09,   2.30186e-17,   4.59282e-25,
     8.84237e-02,   1.76428e-09,   3.52021e-17,   7.02374e-25,
     1.78786e-01,   3.56725e-09,   7.11761e-17,   1.42015e-24,
     4.36404e-01,   8.70740e-09,   1.73736e-16,   3.46648e-24,
     1.63796e+00,   3.26816e-08,   6.52083e-16,   1.30108e-23,
     8.50817e+00,   1.69760e-07,   3.38716e-15,   6.75828e-23,
     3.92751e+01,   7.83641e-07,   1.56357e-14,   3.11973e-22,
     1.41433e+02,   2.82196e-06,   5.63055e-14,   1.12344e-21,
     3.83709e+02,   7.65599e-06,   1.52757e-13,   3.04791e-21,
     7.70411e+02,   1.53717e-05,   3.06706e-13,   6.11959e-21,
     1.16399e+03,   2.32246e-05,   4.63392e-13,   9.24589e-21,
     1.37566e+03,   2.74481e-05,   5.47662e-13,   1.09273e-20,
     1.33070e+03,   2.65509e-05,   5.29761e-13,   1.05701e-20,
     1.09978e+03,   2.19435e-05,   4.37830e-13,   8.73585e-21,
     8.05638e+02,   1.60746e-05,   3.20730e-13,   6.39941e-21,
     5.38690e+02,   1.07483e-05,   2.14456e-13,   4.27897e-21,
     3.36338e+02,   6.71083e-06,   1.33899e-13,   2.67163e-21,
     1.99460e+02,   3.97975e-06,   7.94065e-14,   1.58437e-21,
     1.13787e+02,   2.27035e-06,   4.52995e-14,   9.03844e-22,
     6.30411e+01,   1.25784e-06,   2.50971e-14,   5.00753e-22,
     3.41529e+01,   6.81441e-07,   1.35965e-14,   2.71286e-22,
     1.81893e+01,   3.62924e-07,   7.24128e-15,   1.44483e-22  };


  itab0 = Nmom * NTd * iSN;
  itab  = 0;
  for(imom = 0; imom < Nmom; imom++) {
    for(iTd = 0; iTd < NTd; iTd++) {
      my_rates->SN0_kpSiM     [itab0] = C25_kpSiM     [itab];
      my_rates->SN0_kpFeM     [itab0] = C25_kpFeM     [itab];
      my_rates->SN0_kpMg2SiO4 [itab0] = C25_kpMg2SiO4 [itab];
      my_rates->SN0_kpMgSiO3  [itab0] = C25_kpMgSiO3  [itab];
      my_rates->SN0_kpAC      [itab0] = C25_kpAC      [itab];
      my_rates->SN0_kpSiO2D   [itab0] = C25_kpSiO2D   [itab];
      my_rates->SN0_kpMgO     [itab0] = C25_kpMgO     [itab];
      my_rates->SN0_kpFeS     [itab0] = C25_kpFeS     [itab];
      my_rates->SN0_kpAl2O3   [itab0] = C25_kpAl2O3   [itab];
      itab0++;
      itab ++;
    }
  }

  return SUCCESS;
}
