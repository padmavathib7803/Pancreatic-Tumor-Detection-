import numpy as np
from Global_Vars import Global_Vars
# from Model_Densenet import Model_Nested_Densenet


def Obj_fun_CLS(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    Batch_size = 16
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            # Train_Data = Feat[:learnperc, :]
            # Train_Target = Target[:learnperc, :]
            # Test_Data = Feat[learnperc:, :]
            # Test_Target = Target[learnperc:, :]
            # sol = np.round(Soln[i]).astype('uint8')
            # Eval = Model_Nested_Densenet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size, sol)
            # Fitn[i] = (1 / Eval[13]) + Eval[9] + Eval[11]  # (1 / MCC) + FNR + FDR
            Fitn[i] = (1 / MCC) + FNR + FDR
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        # Train_Data = Feat[:learnperc, :]
        # Train_Target = Target[:learnperc, :]
        # Test_Data = Feat[learnperc:, :]
        # Test_Target = Target[learnperc:, :]
        # Eval = Model_Nested_Densenet(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size, sol)
        # Fitn = (1 / Eval[13]) + Eval[9] + Eval[11]  # (1 / MCC) + FNR + FDR
        Fitn =  (1 / MCC) + FNR + FDR
        return Fitn
