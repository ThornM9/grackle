//Definition of function which saves all rates calculated in initialize_chemistry_data.c to various .txt files.
//This is used for testing purposes.
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include "grackle_macros.h"
#include "grackle_types.h"
#include "grackle_chemistry_data.h"

//Function definition.
int writeRates(chemistry_data *my_chemistry, chemistry_data_storage *my_rates) {
    FILE *fp;
    char fileExtension[5] = ".txt";

    //* Write k1-k58 rates.
    char fileName1[100] = "k1-k58_rates_";
    strcat(fileName1, fileExtension);
    fp = fopen(fileName1, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        fprintf(fp, "%d,", my_rates->k1[line]);
        fprintf(fp, "%d,", my_rates->k2[line]);
        fprintf(fp, "%d,", my_rates->k3[line]);
        fprintf(fp, "%d,", my_rates->k4[line]);
        fprintf(fp, "%d,", my_rates->k5[line]);
        fprintf(fp, "%d,", my_rates->k6[line]);
        fprintf(fp, "%d,", my_rates->k7[line]);
        fprintf(fp, "%d,", my_rates->k8[line]);
        fprintf(fp, "%d,", my_rates->k9[line]);
        fprintf(fp, "%d,", my_rates->k10[line]);
        fprintf(fp, "%d,", my_rates->k11[line]);
        fprintf(fp, "%d,", my_rates->k12[line]);
        fprintf(fp, "%d,", my_rates->k13[line]);
        fprintf(fp, "%d,", my_rates->k14[line]);
        fprintf(fp, "%d,", my_rates->k15[line]);
        fprintf(fp, "%d,", my_rates->k16[line]);
        fprintf(fp, "%d,", my_rates->k17[line]);
        fprintf(fp, "%d,", my_rates->k18[line]);
        fprintf(fp, "%d,", my_rates->k19[line]);
        fprintf(fp, "%d,", my_rates->k20[line]);
        fprintf(fp, "%d,", my_rates->k21[line]);
        fprintf(fp, "%d,", my_rates->k22[line]);
        fprintf(fp, "%d,", my_rates->k23[line]);
        fprintf(fp, "%d,", my_rates->k50[line]);
        fprintf(fp, "%d,", my_rates->k51[line]);
        fprintf(fp, "%d,", my_rates->k52[line]);
        fprintf(fp, "%d,", my_rates->k53[line]);
        fprintf(fp, "%d,", my_rates->k54[line]);
        fprintf(fp, "%d,", my_rates->k55[line]);
        fprintf(fp, "%d,", my_rates->k56[line]);
        fprintf(fp, "%d,", my_rates->k57[line]);
        fprintf(fp, "%d,", my_rates->k58[line]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write H2 formation heating terms.
    char fileName2[100] = "H2formHeating_rates_";
    strcat(fileName2, fileExtension);
    fp = fopen(fileName2, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        fprintf(fp, "%d,", my_rates->n_cr_n[line]);
        fprintf(fp, "%d,", my_rates->n_cr_d1[line]);
        fprintf(fp, "%d,", my_rates->n_cr_d2[line]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write cooling and heating rates.
    char fileName3[100] = "coolingAndHeating_rates_";
    strcat(fileName3, fileExtension);
    fp = fopen(fileName3, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        fprintf(fp, "%d,", my_rates->ceHI[line]);
        fprintf(fp, "%d,", my_rates->ceHeI[line]);
        fprintf(fp, "%d,", my_rates->ceHeII[line]);
        fprintf(fp, "%d,", my_rates->ciHI[line]);
        fprintf(fp, "%d,", my_rates->ciHeI[line]);
        fprintf(fp, "%d,", my_rates->ciHI[line]);
        fprintf(fp, "%d,", my_rates->ciHeIS[line]);
        fprintf(fp, "%d,", my_rates->ciHeII[line]);
        fprintf(fp, "%d,", my_rates->reHII[line]);
        fprintf(fp, "%d,", my_rates->reHeII1[line]);
        fprintf(fp, "%d,", my_rates->reHeII2[line]);
        fprintf(fp, "%d,", my_rates->reHeIII[line]);
        fprintf(fp, "%d,", my_rates->brem[line]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write molecular hydrogen cooling rates.
    char fileName4[100] = "molecHydrogenCooling_rates_";
    strcat(fileName4, fileExtension);
    fp = fopen(fileName4, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        fprintf(fp, "%d,", my_rates->hyd01k[line]);
        fprintf(fp, "%d,", my_rates->h2k01[line]);
        fprintf(fp, "%d,", my_rates->vibh[line]);
        fprintf(fp, "%d,", my_rates->roth[line]);
        fprintf(fp, "%d,", my_rates->rotl[line]);
        fprintf(fp, "%d,", my_rates->HDlte[line]);
        fprintf(fp, "%d,", my_rates->HDlow[line]);
        fprintf(fp, "%d,", my_rates->cieco[line]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write low density rates.
    char fileName5[100] = "lowDensity_rates_";
    strcat(fileName5, fileExtension);
    fp = fopen(fileName5, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        fprintf(fp, "%d,", my_rates->GAHI[line]);
        fprintf(fp, "%d,", my_rates->GAH2[line]);
        fprintf(fp, "%d,", my_rates->GAHe[line]);
        fprintf(fp, "%d,", my_rates->GAHI[line]);
        fprintf(fp, "%d,", my_rates->GAHp[line]);
        fprintf(fp, "%d,", my_rates->GAel[line]);
        fprintf(fp, "%d,", my_rates->H2LTE[line]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write k13dd.
    char fileName6[100] = "k13dd_";
    strcat(fileName6, fileExtension);
    fp = fopen(fileName6, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        for (int col = 0; col < 14; col++) {
            fprintf(fp, "%d,", my_rates->k13dd[line + col*my_chemistry->NumberOfTemperatureBins]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    //*Write h2dust.
    char fileName7[100] = "h2dust_";
    strcat(fileName7, fileExtension);
    fp = fopen(fileName7, "w");
    for (int line = 0; line < my_chemistry->NumberOfTemperatureBins; line++) {
        for (int col = 0; col < my_chemistry->NumberOfDustTemperatureBins; col++) {
            fprintf(fp, "%d,", my_rates->h2dust[line + col*my_chemistry->NumberOfTemperatureBins]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return SUCCESS;
}