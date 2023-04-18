/***********************************************************************
/
/ Grackle function prototypes
/
/
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/
/ The full license is in the file LICENSE, distributed with this 
/ software.
************************************************************************/

#ifndef __GRACKLE_H__
#define __GRACKLE_H__

#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include <stddef.h>

extern int grackle_verbose;

extern chemistry_data *grackle_data;
extern chemistry_data_storage grackle_rates;

double get_velocity_units(code_units *my_units);

void set_velocity_units(code_units *my_units);

double get_temperature_units(code_units *my_units);

int set_default_chemistry_parameters(chemistry_data *my_grackle);

chemistry_data _set_default_chemistry_parameters(void);

int initialize_chemistry_data(code_units *my_units);

int _initialize_chemistry_data(chemistry_data *my_chemistry, 
                               chemistry_data_storage *my_rates,
                               code_units *my_units);

int* local_chemistry_data_access_int(chemistry_data* my_chemistry,
                                     const char* param_name);
double* local_chemistry_data_access_double(chemistry_data* my_chemistry,
                                           const char* param_name);
char** local_chemistry_data_access_string(chemistry_data* my_chemistry,
                                          const char* param_name);

const char* param_name_int(size_t i);
const char* param_name_double(size_t i);
const char* param_name_string(size_t i);

size_t grackle_num_params(const char* type_name);

int solve_chemistry(code_units *my_units,
                    grackle_field_data *my_fields,
                    double dt_value);

int local_solve_chemistry(chemistry_data *my_chemistry,
                          chemistry_data_storage *my_rates,
                          code_units *my_units,
                          grackle_field_data *my_fields,
                          double dt_value);

int _solve_chemistry(chemistry_data *my_chemistry,
                     chemistry_data_storage *my_rates,
                     code_units *my_units, double dt_value, double dx_value,
                     int grid_rank, int *grid_dimension,
                     int *grid_start, int *grid_end,
                     gr_float *density, gr_float *internal_energy,
                     gr_float *x_velocity, gr_float *y_velocity, gr_float *z_velocity,
                     gr_float *HI_density, gr_float *HII_density, gr_float *HM_density,
                     gr_float *HeI_density, gr_float *HeII_density, gr_float *HeIII_density,
                     gr_float *H2I_density, gr_float *H2II_density,
                     gr_float *DI_density, gr_float *DII_density, gr_float *HDI_density,
                     gr_float *DM_density, gr_float *HDII_density, gr_float *HeHII_density,
                     gr_float *CI_density,  gr_float *CII_density,  gr_float *CO_density,  gr_float *CO2_density,
                     gr_float *OI_density,  gr_float *OH_density,  gr_float *H2O_density,  gr_float *O2_density,
                     gr_float *SiI_density,  gr_float *SiOI_density,  gr_float *SiO2I_density,
                     gr_float *CH_density,  gr_float *CH2_density,  gr_float *COII_density,  gr_float *OII_density,
                     gr_float *OHII_density,  gr_float *H2OII_density,  gr_float *H3OII_density,  gr_float *O2II_density,
                     gr_float *Mg_density,  gr_float *Al_density,  gr_float *S_density,  gr_float *Fe_density,
                     gr_float *SiM_density,  gr_float *FeM_density,  gr_float *Mg2SiO4_density,  gr_float *MgSiO3_density,  gr_float *Fe3O4_density,
                     gr_float *AC_density,  gr_float *SiO2D_density,  gr_float *MgO_density,  gr_float *FeS_density,  gr_float *Al2O3_density,
                     gr_float *reforg, gr_float *volorg, gr_float *H2Oice,
                     gr_float *e_density, gr_float *metal_density, gr_float *dust_density,
                     gr_float *metal_loc, gr_float *metal_C13, gr_float *metal_C20, gr_float *metal_C25, gr_float *metal_C30,
                     gr_float *metal_F13, gr_float *metal_F15, gr_float *metal_F50, gr_float *metal_F80,
                     gr_float *metal_P170, gr_float *metal_P200, gr_float *metal_Y19,
                     gr_float *volumetric_heating_rate, gr_float *specific_heating_rate,
                     gr_float *RT_heating_rate, gr_float *RT_HI_ionization_rate, gr_float *RT_HeI_ionization_rate,
                     gr_float *RT_HeII_ionization_rate, gr_float *RT_H2_dissociation_rate, gr_float *RT_HDI_dissociation_rate, 
                     gr_float *RT_CI_ionization_rate, gr_float *RT_OI_ionization_rate,
                     gr_float *RT_CO_dissociation_rate, gr_float *RT_OH_dissociation_rate, gr_float *RT_H2O_dissociation_rate, 
                     gr_float *H2_self_shielding_length) __attribute__ ((deprecated));

int calculate_cooling_time(code_units *my_units,
                           grackle_field_data *my_fields,
                           gr_float *cooling_time);

int local_calculate_cooling_time(chemistry_data *my_chemistry,
                                 chemistry_data_storage *my_rates,
                                 code_units *my_units,
                                 grackle_field_data *my_fields,
                                 gr_float *cooling_time);

int _calculate_cooling_time(chemistry_data *my_chemistry,
                            chemistry_data_storage *my_rates,
                            code_units *my_units,
                            int grid_rank, int *grid_dimension,
                            int *grid_start, int *grid_end,
                            gr_float *density, gr_float *internal_energy,
                            gr_float *x_velocity, gr_float *y_velocity, gr_float *z_velocity,
                            gr_float *HI_density, gr_float *HII_density, gr_float *HM_density,
                            gr_float *HeI_density, gr_float *HeII_density, gr_float *HeIII_density,
                            gr_float *H2I_density, gr_float *H2II_density,
                            gr_float *DI_density, gr_float *DII_density, gr_float *HDI_density,
                            gr_float *DM_density, gr_float *HDII_density, gr_float *HeHII_density,
                            gr_float *CI_density,  gr_float *CII_density,  gr_float *CO_density,  gr_float *CO2_density,
                            gr_float *OI_density,  gr_float *OH_density,  gr_float *H2O_density,  gr_float *O2_density,
                            gr_float *SiI_density,  gr_float *SiOI_density,  gr_float *SiO2I_density,
                            gr_float *CH_density,  gr_float *CH2_density,  gr_float *COII_density,  gr_float *OII_density,
                            gr_float *OHII_density,  gr_float *H2OII_density,  gr_float *H3OII_density,  gr_float *O2II_density,
                            gr_float *Mg_density,  gr_float *Al_density,  gr_float *S_density,  gr_float *Fe_density,
                            gr_float *SiM_density,  gr_float *FeM_density,  gr_float *Mg2SiO4_density,  gr_float *MgSiO3_density,  gr_float *Fe3O4_density,
                            gr_float *AC_density,  gr_float *SiO2D_density,  gr_float *MgO_density,  gr_float *FeS_density,  gr_float *Al2O3_density,
                            gr_float *reforg, gr_float *volorg, gr_float *H2Oice,
                            gr_float *e_density, gr_float *metal_density, gr_float *dust_density,
                            gr_float *metal_loc, gr_float *metal_C13, gr_float *metal_C20, gr_float *metal_C25, gr_float *metal_C30,
                            gr_float *metal_F13, gr_float *metal_F15, gr_float *metal_F50, gr_float *metal_F80,
                            gr_float *metal_P170, gr_float *metal_P200, gr_float *metal_Y19,
                            gr_float *cooling_time, gr_float *RT_heating_rate,
                            gr_float *volumetric_heating_rate, gr_float *specific_heating_rate) __attribute__ ((deprecated));

int calculate_dust_temperature(code_units *my_units,
                               grackle_field_data *my_fields,
                               gr_float *dust_temperature);

int local_calculate_dust_temperature(chemistry_data *my_chemistry,
                                     chemistry_data_storage *my_rates,
                                     code_units *my_units,
                                     grackle_field_data *my_fields,
                                     gr_float *dust_temperature);

int calculate_gamma(code_units *my_units,
                    grackle_field_data *my_fields,
                    gr_float *my_gamma);

int local_calculate_gamma(chemistry_data *my_chemistry,
                          chemistry_data_storage *my_rates,
                          code_units *my_units,
                          grackle_field_data *my_fields,
                          gr_float *my_gamma);

int _calculate_gamma(chemistry_data *my_chemistry,
                     chemistry_data_storage *my_rates,
                     code_units *my_units,
                     int grid_rank, int *grid_dimension,
                     int *grid_start, int *grid_end,
                     gr_float *density, gr_float *internal_energy,
                     gr_float *HI_density, gr_float *HII_density, gr_float *HM_density,
                     gr_float *HeI_density, gr_float *HeII_density, gr_float *HeIII_density,
                     gr_float *H2I_density, gr_float *H2II_density,
                     gr_float *DI_density, gr_float *DII_density, gr_float *HDI_density,
                     gr_float *DM_density, gr_float *HDII_density, gr_float *HeHII_density,
                     gr_float *CI_density,  gr_float *CII_density,  gr_float *CO_density,  gr_float *CO2_density,
                     gr_float *OI_density,  gr_float *OH_density,  gr_float *H2O_density,  gr_float *O2_density,
                     gr_float *SiI_density,  gr_float *SiOI_density,  gr_float *SiO2I_density,
                     gr_float *CH_density,  gr_float *CH2_density,  gr_float *COII_density,  gr_float *OII_density,
                     gr_float *OHII_density,  gr_float *H2OII_density,  gr_float *H3OII_density,  gr_float *O2II_density,
                     gr_float *Mg_density,  gr_float *Al_density,  gr_float *S_density,  gr_float *Fe_density,
                     gr_float *SiM_density,  gr_float *FeM_density,  gr_float *Mg2SiO4_density,  gr_float *MgSiO3_density,  gr_float *Fe3O4_density,
                     gr_float *AC_density,  gr_float *SiO2D_density,  gr_float *MgO_density,  gr_float *FeS_density,  gr_float *Al2O3_density,
                     gr_float *reforg, gr_float *volorg, gr_float *H2Oice,
                     gr_float *e_density, gr_float *metal_density,
                     gr_float *my_gamma) __attribute__ ((deprecated));

int calculate_pressure(code_units *my_units,
                       grackle_field_data *my_fields,
                       gr_float *pressure);

int local_calculate_pressure(chemistry_data *my_chemistry,
                             chemistry_data_storage *my_rates,
                             code_units *my_units,
                             grackle_field_data *my_fields,
                             gr_float *pressure);

int _calculate_pressure(chemistry_data *my_chemistry,
                        chemistry_data_storage *my_rates,
                        code_units *my_units,
                        int grid_rank, int *grid_dimension,
                        int *grid_start, int *grid_end,
                        gr_float *density, gr_float *internal_energy,
                        gr_float *HI_density, gr_float *HII_density, gr_float *HM_density,
                        gr_float *HeI_density, gr_float *HeII_density, gr_float *HeIII_density,
                        gr_float *H2I_density, gr_float *H2II_density,
                        gr_float *DI_density, gr_float *DII_density, gr_float *HDI_density,
                        gr_float *DM_density, gr_float *HDII_density, gr_float *HeHII_density,
                        gr_float *CI_density,  gr_float *CII_density,  gr_float *CO_density,  gr_float *CO2_density,
                        gr_float *OI_density,  gr_float *OH_density,  gr_float *H2O_density,  gr_float *O2_density,
                        gr_float *SiI_density,  gr_float *SiOI_density,  gr_float *SiO2I_density,
                        gr_float *CH_density,  gr_float *CH2_density,  gr_float *COII_density,  gr_float *OII_density,
                        gr_float *OHII_density,  gr_float *H2OII_density,  gr_float *H3OII_density,  gr_float *O2II_density,
                        gr_float *Mg_density,  gr_float *Al_density,  gr_float *S_density,  gr_float *Fe_density,
                        gr_float *SiM_density,  gr_float *FeM_density,  gr_float *Mg2SiO4_density,  gr_float *MgSiO3_density,  gr_float *Fe3O4_density,
                        gr_float *AC_density,  gr_float *SiO2D_density,  gr_float *MgO_density,  gr_float *FeS_density,  gr_float *Al2O3_density,
                        gr_float *reforg, gr_float *volorg, gr_float *H2Oice,
                        gr_float *e_density, gr_float *metal_density,
                        gr_float *pressure) __attribute__ ((deprecated));

int calculate_temperature(code_units *my_units,
                          grackle_field_data *my_fields,
                          gr_float *temperature);

int local_calculate_temperature(chemistry_data *my_chemistry,
                                chemistry_data_storage *my_rates,
                                code_units *my_units,
                                grackle_field_data *my_fields,
                                gr_float *temperature);

int _calculate_temperature(chemistry_data *my_chemistry,
                           chemistry_data_storage *my_rates,
                           code_units *my_units,
                           int grid_rank, int *grid_dimension,
                           int *grid_start, int *grid_end,
                           gr_float *density, gr_float *internal_energy,
                           gr_float *HI_density, gr_float *HII_density, gr_float *HM_density,
                           gr_float *HeI_density, gr_float *HeII_density, gr_float *HeIII_density,
                           gr_float *H2I_density, gr_float *H2II_density,
                           gr_float *DI_density, gr_float *DII_density, gr_float *HDI_density,
                           gr_float *DM_density, gr_float *HDII_density, gr_float *HeHII_density,
                           gr_float *CI_density,  gr_float *CII_density,  gr_float *CO_density,  gr_float *CO2_density,
                           gr_float *OI_density,  gr_float *OH_density,  gr_float *H2O_density,  gr_float *O2_density,
                           gr_float *SiI_density,  gr_float *SiOI_density,  gr_float *SiO2I_density,
                           gr_float *CH_density,  gr_float *CH2_density,  gr_float *COII_density,  gr_float *OII_density,
                           gr_float *OHII_density,  gr_float *H2OII_density,  gr_float *H3OII_density,  gr_float *O2II_density,
                           gr_float *Mg_density,  gr_float *Al_density,  gr_float *S_density,  gr_float *Fe_density,
                           gr_float *SiM_density,  gr_float *FeM_density,  gr_float *Mg2SiO4_density,  gr_float *MgSiO3_density,  gr_float *Fe3O4_density,
                           gr_float *AC_density,  gr_float *SiO2D_density,  gr_float *MgO_density,  gr_float *FeS_density,  gr_float *Al2O3_density,
                           gr_float *reforg, gr_float *volorg, gr_float *H2Oice,
                           gr_float *e_density, gr_float *metal_density,
                           gr_float *temperature) __attribute__ ((deprecated));

int _free_chemistry_data(chemistry_data *my_chemistry, chemistry_data_storage *my_rates);

grackle_version get_grackle_version(void);

#endif
