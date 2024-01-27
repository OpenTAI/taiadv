from . import RST, UAT, TRADES, MART, MMA, BAT, ADVInterp, FeaScatter, Sense
from . import JARN_AT, Dynamic, AWP, Overfitting, ATHE, PreTrain, SAT
from . import RobustWRN, SAT_VGG, SAT_DARTS, SAT_DENSENET
from . import FASTAT_IN1K, Engstrom2019Robustness, Debenedetti2022Light_XCiT_S12, Salman2020Do_R50
from . import CleanWRN, CleanWRN_Smoothing, SATWRN, SATWRN_Smoothing

defence_options = {
    "RST": RST.DefenseRST,
    "UAT": UAT.DefenseUAT,
    "TRADES": TRADES.DefenseTRADES,
    "MART": MART.DefenseMART,
    "MMA": MMA.DefenseMMA,
    "BAT": BAT.DefenseBAT,
    "ADVInterp": ADVInterp.DefenseADVInterp,
    "FeaScatter": FeaScatter.DefenseFeaScatter,
    "Sense": Sense.DefenseSense,
    "JARN_AT": JARN_AT.DefenseJARN_AT,
    "Dynamic": Dynamic.DefenseDynamic,
    "AWP": AWP.DefenseAWP,
    "Overfitting": Overfitting.DefenseOverfitting,
    "ATHE": ATHE.DefenseATHE,
    "PreTrain": PreTrain.DefensePreTrain,
    "SAT": SAT.DefenseSAT,
    "RobustWRN": RobustWRN.DefenseRobustWRN,
    "SAT_VGG": SAT_VGG.DefenseSATvgg,
    "SAT_DARTS": SAT_DARTS.DefenseSATdarts,
    "SAT_DENSENET": SAT_DENSENET.DefenseSATdensenet,
    "FASTAT_IN1K": FASTAT_IN1K.DefenseFastATIN1K,
    "Engstrom2019Robustness": Engstrom2019Robustness.Engstrom2019Robustness,
    "Salman2020Do_R50": Salman2020Do_R50.Salman2020Do_R50,
    "Debenedetti2022Light_XCiT_S12": Debenedetti2022Light_XCiT_S12.Debenedetti2022Light_XCiT_S12,
    "CleanWRN": CleanWRN.CleanWRN,
    "CleanWRN_Smoothing": CleanWRN_Smoothing.CleanWRN_Smoothing,
    "SATWRN": SATWRN.SATWRN,
    "SATWRN_Smoothing": SATWRN_Smoothing.SATWRN_Smoothing,
}
