#%%
import tenseal as ts
import math
import logging
import os
import pandas as pd
# %%Params to investigate
modulus_and_coeff = [[4096,109],[8192, 218]]
Both_Ends = [30,35,40,45,50,55,60]
All_Combos = dict()

#%%Set the directory to the file location
filename = os.path.dirname(__file__)
os.chdir(filename)
print(os.getcwd())

#And also name for logging
name = __file__.split('\\')[-1].split('.')[0]
print(name)

#%%Define logger
logging.basicConfig(filename=f'{name}_logs.txt',
                    filemode='w',
                    format='Time: %(asctime)s,%(msecs)02d, Message: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger()

#Add stream handler for logging
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# %%Find combinations
i = 1
for mac in modulus_and_coeff:
    for Ends in Both_Ends:
        poly_deg = mac[0]
        total_budget = mac[1]
        minimum = int(math.log2(poly_deg)) + 1
        mids_count = 5
        valid_tokens = []
        
        for mid_val in range(minimum, Ends):  # Full to 60, ignore budget for increment
            token = [Ends] + [mid_val] * mids_count + [Ends]

            current_sum = sum(token)
            if current_sum > total_budget:
                break  # Stop if per-token exceeds total
            
            try:
                temp = ts.context(ts.SCHEME_TYPE.CKKS, 
                                  poly_modulus_degree=poly_deg, coeff_mod_bit_sizes=token)
                valid_tokens.append(token)
                All_Combos[i] = [mac[0], token]
                i += 1
            except Exception as e:
                pass
        
        if valid_tokens:
            logger.info(f"Valid sets for {mac}, Ends={Ends} (each sum <= {total_budget}):")
            for tok in valid_tokens:
                logger.info(tok)
        else:
            logger.info(f"No valid tokens for {mac}, {Ends}")

#%%
All_dat = pd.DataFrame.from_dict(All_Combos).T
All_dat
# %%
All_dat.to_excel('Modulus_and_Coefficients_combo_3.xlsx')
# %%
