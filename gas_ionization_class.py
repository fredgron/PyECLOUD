#-Begin-preamble-------------------------------------------------------
#
#                           CERN
#
#     European Organization for Nuclear Research
#
#
#     This file is part of the code:
#
#                   PyECLOUD Version 8.7.0
#
#
#     Main author:          Giovanni IADAROLA
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.iadarola@cern.ch
#
#     Contributors:         Eleonora Belli
#                           Philipp Dijkstal
#                           Lorenzo Giacomel
#                           Lotta Mether
#                           Annalisa Romano
#                           Giovanni Rumolo
#                           Eric Wulff
#                           Fredrik Grønvold
#
#
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other
#     appropriate  legal  protection  of  this  computer program and
#     associated documentation reserved  in  all  countries  of  the
#     world.
#
#     Organizations collaborating with CERN may receive this program
#     and documentation freely and without charge.
#
#     CERN undertakes no obligation  for  the  maintenance  of  this
#     program,  nor responsibility for its correctness,  and accepts
#     no liability whatsoever resulting from its use.
#
#     Program  and documentation are provided solely for the use  of
#     the organization to which they are distributed.
#
#     This program  may  not  be  copied  or  otherwise  distributed
#     without  permission. This message must be retained on this and
#     any other authorized copies.
#
#     The material cannot be sold. CERN should be  given  credit  in
#     all references.
#
#-End-preamble---------------------------------------------------------


from numpy.random import rand
from numpy.random import randn
from numpy import *
from scipy.constants import c, k, e, epsilon_0
import PyECLOUD.BassErsk as BE

class residual_gas_ionization:

    def __init__(self, unif_frac, P_nTorr, sigma_ion_MBarn, Temp_K, chamb, E_init_ion, EFI_th, EFI_sz0, EFI_bsp, EFI_prob, flag_lifetime_hist = False):

        print('Start res. gas ioniz. init.')
        self.unif_frac = unif_frac
        self.P_nTorr = P_nTorr
        self.sigma_ion_MBarn = sigma_ion_MBarn
        self.Temp_K = Temp_K
#         self.sigmax = sigmax
#         self.sigmay = sigmay
        self.chamb = chamb
        self.E_init_ion = E_init_ion

        self.EFI_th = EFI_th    #If EFI_th = -1, then EFI_flag = False ??
        self.bsp = EFI_bsp      #Bunch spacing
        self.sigmaz = EFI_sz0   #Conversion to match sz0 in FASTION?
        self.Efield_flag = False
        self.EFI_prob = EFI_prob
        self.sx0fion = 0
        self.sy0fion = 0

        if self.bsp != -1 and self.EFI_th != -1 and self.sigmaz != -1:
            self.use_EFI_flag = True 

        elif self.bsp == -1 and self.EFI_th == -1 and self.sigmaz == -1:
            self.use_EFI_flag = False

        else:
            raise ValueError("To use EFI feature all EFI parameters have to be defined")


#         self.x_beam_pos = x_beam_pos
#         self.y_beam_pos = y_beam_pos

        self.flag_lifetime_hist = flag_lifetime_hist

        print('Done res. gas ioniz. init.')

    #@profile
    def generate(self, MP_e, lambda_t, Dt, sigmax, sigmay, x_beam_pos=0., y_beam_pos=0., bnum = None):

        mass = MP_e.mass

        v0 = -sqrt(2. * (self.E_init_ion / 3.) * e / mass)

        P_Pa = self.P_nTorr * 133.32e-9                 # Unit conversion to [Pa]
        sigma_ion_mq = self.sigma_ion_MBarn * 1e-22     # Unit conversion to [m]
        n_gas = P_Pa / (k * self.Temp_K)    # Get the gas density

        if self.use_EFI_flag == False:
            self.scattering_ionisation(MP_e,n_gas, sigma_ion_mq, lambda_t, Dt, sigmax, sigmay, x_beam_pos, y_beam_pos, v0)

        else:
            if bnum == None: raise ValueError("\'bnum\' must be known to use EFI feature")
            
            bunchnum = bnum

            if bunchnum == 0:
                maxe = 0
                xionlim2 = 0
                yionlim = 0

                print("Starting search...")
                #--- Search along y-axis ---#
                xtst = 0.
                flag1 = True
                for qq in range(1000, 0, -1):
                    ytst = qq*10./1000. *sigmax
                    xoffc = 0.
                    yoffc = 0.
                    xin = xtst - xoffc
                    yin = ytst - yoffc
                    
                    den = sqrt((40.*sigmax)**2+(xin*xin+yin*yin))      #To make it fit with E-fieldC
                    w1 = 40.*sigmax/den                                #To make it fit with E-fieldC

                    Ex, Ey = BE.BassErsk(xin,yin,sigmax, sigmay)                              # Calculate beam electric field
                    
                    Ex = w1*Ex      #To make it fit with E-fieldC
                    Ey = w1*Ey      #To make it fit with E-fieldC
                    
                    elec_norm = (lambda_t*Dt*c*e)/0.3/self.sigmaz/sqrt(2*pi)/self.EFI_th/1e9*sqrt(Ey**2)    # Normalise electric field in y by field threshold

                    if elec_norm > 1.0 and flag1:
                        yionlim = ytst              # Set limit of ionisation area in y
                        flag1 = False               # Activate flag to ensure that yionlim does not get overwritten by the loop
                        #print('Hello from first scan!, qq = %d' %qq) 

                    if elec_norm > maxe: 
                        maxe = elec_norm    # Record the highest value of the normalised beam electric field
                        ymax = ytst         # Record the y pos of the highest normalised electric field

                print("First search completed")
                
                #--- Search along x-axis ---#
                ytst = ymax
                for pp in range(1000, 0, -1):
                    xtst = pp*10./1000. *sigmax
                    xoffc = 0.
                    yoffc = 0.
                    xin = xtst - xoffc
                    yin = ytst - yoffc

                    den = sqrt((40.*sigmax)**2+(xin*xin+yin*yin))      #To make it fit with E-fieldC
                    w1 = 40.*sigmax/den                                #To make it fit with E-fieldC
                    
                    Ex, Ey = BE.BassErsk(xin,yin,sigmax, sigmay)                                      # Calculate beam electric field
                    
                    Ex = w1*Ex      #To make it fit with E-fieldC
                    Ey = w1*Ey      #To make it fit with E-fieldC
                    
                    elec_norm = (lambda_t*Dt*c*e)/0.3/self.sigmaz/sqrt(2*pi)/self.EFI_th/1e9*sqrt(Ey**2 + Ex**2)    # Normalise entire beam electric field by the electric field threshold
                    
                    # if pp == 216:
                    #     print("pp: %i"%pp)
                    #     print("E_norm: %.10f"%elec_norm)
                    #     print("Ex: %.10f \tEy: %.10f"%(Ex*(2*sqrt(2*pi)*epsilon_0), Ey*(2*sqrt(2*pi)*epsilon_0)))
                    #     print("%.8e\t%.8e\t%.8e"%(lambda_t*Dt*c*e,(lambda_t*Dt*c*e)/0.3,(lambda_t*Dt*c*e)/0.3/self.EFI_th))
                        

                    if elec_norm > 1.0:
                        print("pp: %i"%pp)
                        print("E_norm: %.10f"%elec_norm)
                        print("Ex: %.10f \tEy: %.10f"%(Ex*(2*sqrt(2*pi)*epsilon_0), Ey*(2*sqrt(2*pi)*epsilon_0)))

                        # with open(("./Efield_comparison/EfieldPy_E_norm_sx_%.4fum_sy_%fum.dat"%(sigmax,sigmay)),"a") as file:
                        #     file.write("%lf\t%lf\t%lf\t%i"%(Ex*(2*sqrt(2*pi)*epsilon_0), Ey*(2*sqrt(2*pi)*epsilon_0), elec_norm, pp))

                        xionlim2 = xtst     # Set limit of ionisation area in x
                        break               # End search and exit for-loop
                
                print("Second search completed")

                self.sx0fion = xionlim2          # Set sigmax of ionisation area to limit found in search
                self.sy0fion = yionlim           # Set sigmay of ionisation area to limit found in search

                if xionlim2>1.e-7:          # If the ionisation area dimension in x exceeds 0.1 um
                    self.Efield_flag = True      # Actiavte field ionisation flag
                    print("Beam electric field fulfils requirements for field ionisation")
                else:
                    self.Efield_flag = False     # Deactivate field ionisation flag
                    print("Beam electric field does not fulfil requirements for field ionisation")
            
                print("Completed E-field scan")

            if not self.Efield_flag:
                
                self.scattering_ionisation(MP_e,n_gas, sigma_ion_mq, lambda_t, Dt, sigmax, sigmay, x_beam_pos, y_beam_pos, v0)

            else:
                self.efield_ionisation(MP_e, bunchnum, n_gas, sigma_ion_mq, lambda_t, Dt, self.sx0fion, self.sy0fion, x_beam_pos, y_beam_pos, v0)                

        #if Nint_new_MP > 0:
        return MP_e

    def scattering_ionisation(self, MP_e, n_gas, sigma_ion_mq, lambda_t, Dt, sigmax, sigmay, x_beam_pos, y_beam_pos, v0):

        k_ion = n_gas * sigma_ion_mq * c
        DNel = k_ion * lambda_t * Dt
        N_new_MP = DNel / MP_e.nel_mp_ref
        Nint_new_MP = floor(N_new_MP)
        rest = N_new_MP - Nint_new_MP
        Nint_new_MP = int(Nint_new_MP) + int(rand() < rest)
        #print('DNel:%.3e\n Nint_new_MP:%d'%(DNel,Nint_new_MP))
        # print(MP_e.nel_mp_ref)
        # print(Nint_new_MP)

        if Nint_new_MP > 0:
            unif_flag = (rand(Nint_new_MP) < self.unif_frac)
            gauss_flag = ~(unif_flag)

            x_temp = gauss_flag * (sigmax * randn(Nint_new_MP) + x_beam_pos) + self.chamb.x_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))
            y_temp = gauss_flag * (sigmay * randn(Nint_new_MP) + y_beam_pos) + self.chamb.y_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))

            flag_np = self.chamb.is_outside(x_temp, y_temp) # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
            Nout = int(sum(flag_np))

            while(Nout > 0):
                unif_flag1 = unif_flag[flag_np]
                gauss_flag1 = ~(unif_flag1)
                x_temp[flag_np] = gauss_flag1 * (sigmax * randn(Nout) + x_beam_pos) + self.chamb.x_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                y_temp[flag_np] = gauss_flag1 * (sigmay * randn(Nout) + y_beam_pos) + self.chamb.y_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                flag_np = self.chamb.is_outside(x_temp, y_temp)  # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
                Nout = int(sum(flag_np))
        
            # print(shape(x_temp))
            # print(Nint_new_MP)
        
            MP_e.x_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = x_temp # Be careful to the indexing when translating to python
            MP_e.y_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = y_temp
            MP_e.z_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = 0. # randn(Nint_new_MP,1)
            MP_e.vx_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5) # if you note a towards down polarization look here
            MP_e.vy_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
            MP_e.vz_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
            MP_e.nel_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = MP_e.nel_mp_ref
        
            if self.flag_lifetime_hist:
                MP_e.t_last_impact[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = -1

            MP_e.N_mp = int(MP_e.N_mp + Nint_new_MP)


    def efield_ionisation(self, MP_e, bunchnum, n_gas , sigma_ion_mq, lambda_t, Dt, sx0fion, sy0fion, x_beam_pos, y_beam_pos, v0):

        print("E-field Ionisation:  bunch %d"%(bunchnum))

        if bunchnum <= self.EFI_prob: #Ionisation process for the first 10 bunches
                        
            DNel = n_gas*(pi*sx0fion*sy0fion)/self.EFI_prob

            
            if bunchnum == 0:
                MP_e.nel_mp_ref = (MP_e.nel_mp_ref * pi*sx0fion*sy0fion)/(self.EFI_prob*sigma_ion_mq*lambda_t*c*Dt) #This is done to keep the number of MP constant ~500

            N_new_MP = DNel/MP_e.nel_mp_ref

            Nint_new_MP = floor(N_new_MP)
            rest = N_new_MP - Nint_new_MP
            Nint_new_MP = int(Nint_new_MP) + int(rand() < rest)

            #qion = DNel/Nint_new_MP

            # print('DNel:%.3e\n Nint_new_MP:%d'%(DNel,Nint_new_MP))

            if Nint_new_MP > 0:

                unif_flag = (rand(Nint_new_MP) < self.unif_frac)
                gauss_flag = ~(unif_flag)

                randangle = 2*pi*random.uniform(0., 1., Nint_new_MP)
                randradius = sqrt(random.uniform(0., 1., Nint_new_MP))

                x_temp = gauss_flag * (sx0fion * randradius*cos(randangle) + x_beam_pos) + self.chamb.x_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))
                y_temp = gauss_flag * (sy0fion * randradius*sin(randangle) + y_beam_pos) + self.chamb.y_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))

                flag_np = self.chamb.is_outside(x_temp, y_temp) # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
                Nout = int(sum(flag_np))

                while(Nout > 0):
                    unif_flag1 = unif_flag[flag_np]
                    gauss_flag1 = ~(unif_flag1)
                    x_temp[flag_np] = gauss_flag1 * (sx0fion * randradius*cos(randangle) + x_beam_pos) + self.chamb.x_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                    y_temp[flag_np] = gauss_flag1 * (sy0fion * randradius*sin(randangle) + y_beam_pos) + self.chamb.y_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                    flag_np = self.chamb.is_outside(x_temp, y_temp)  # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
                    Nout = int(sum(flag_np))
                
                MP_e.x_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = x_temp # Be careful to the indexing when translating to python
                MP_e.y_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = y_temp
                MP_e.z_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = 0. # randn(Nint_new_MP,1)
                MP_e.vx_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5) # if you note a towards down polarization look here
                MP_e.vy_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
                MP_e.vz_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
                MP_e.nel_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = MP_e.nel_mp_ref
        
            if self.flag_lifetime_hist:
                MP_e.t_last_impact[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = -1

            MP_e.N_mp = int(MP_e.N_mp + Nint_new_MP)

        elif bunchnum > self.EFI_prob: #Ionisation process for the subsequent bunches
                            
            vth = sqrt((8*k*self.Temp_K)/(pi*MP_e.mass))
            DNel = (pi/4)*n_gas*vth*(sx0fion+sy0fion)*self.bsp

            if bunchnum == self.EFI_prob+1:                        
                MP_e.nel_mp_ref = (MP_e.nel_mp_ref*(pi/4)*vth*(sx0fion+sy0fion)*self.bsp*self.EFI_prob)/(pi*sx0fion*sy0fion) #This is done to keep the number of MP constant ~500
            
            N_new_MP = DNel/MP_e.nel_mp_ref

            Nint_new_MP = floor(N_new_MP)
            rest = N_new_MP - Nint_new_MP
            Nint_new_MP = int(Nint_new_MP) + int(rand() < rest)
            
            #qion = DNel/Nint_new_MP

            # print('DNel:%.3e\n Nint_new_MP:%d'%(DNel,Nint_new_MP))

            if Nint_new_MP > 0:

                unif_flag = (rand(Nint_new_MP) < self.unif_frac)
                gauss_flag = ~(unif_flag)
                
                randangle = 2*pi*random.uniform(0., 1., Nint_new_MP)
                randradius = sqrt(random.uniform(0., 1., Nint_new_MP))
                
                r1 = sqrt((sx0fion-1e-7)**2 + randradius*(2*sx0fion-1e-7)*1e-7)
                r2 = sqrt((sy0fion-1e-7)**2 + randradius*(2*sy0fion-1e-7)*1e-7)

                x_temp = gauss_flag * (x_beam_pos + r1*cos(randangle)) + self.chamb.x_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))
                y_temp = gauss_flag * (y_beam_pos + r2*sin(randangle)) + self.chamb.y_aper * unif_flag * (2. * (rand(Nint_new_MP) - 0.5))

                flag_np = self.chamb.is_outside(x_temp, y_temp) # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
                
                Nout = int(sum(flag_np))

                while(Nout > 0):
                    unif_flag1 = unif_flag[flag_np]
                    gauss_flag1 = ~(unif_flag1)
                    x_temp[flag_np] = gauss_flag1 * (x_beam_pos + r1*cos(randangle))  + self.chamb.x_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                    y_temp[flag_np] = gauss_flag1 * (y_beam_pos + r2*sin(randangle))+ self.chamb.y_aper * unif_flag1 * (2 * (rand(Nout) - 0.5))
                    flag_np = self.chamb.is_outside(x_temp, y_temp)  # (((x_temp/x_aper)**2 + (y_temp/y_aper)**2)>=1)
                    Nout = int(sum(flag_np))

                MP_e.x_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = x_temp # Be careful to the indexing when translating to python
                MP_e.y_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = y_temp
                MP_e.z_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = 0. # randn(Nint_new_MP,1)
                MP_e.vx_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5) # if you note a towards down polarization look here
                MP_e.vy_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
                MP_e.vz_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = v0 * (rand() - 0.5)
                MP_e.nel_mp[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = MP_e.nel_mp_ref
            
                if self.flag_lifetime_hist:
                    MP_e.t_last_impact[ MP_e.N_mp: MP_e.N_mp + Nint_new_MP] = -1

                MP_e.N_mp = int(MP_e.N_mp + Nint_new_MP)
