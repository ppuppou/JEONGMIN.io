import pandas as pd
import numpy as np

def infer_fuel_type(engine):
    eng = str(engine).lower()
    
    # 1. 전기차 관련
    if any(x in eng for x in [
        "electric motor", "battery electric", "dual motor", 
        "standard range battery", "kwh"
    ]):
        return "electric"
    
    # 2. 하이브리드
    elif any(x in eng for x in ["hybrid", "mild electric"]):
        return "hybrid"
    
    # 3. 수소차
    elif "hydrogen" in eng:
        return "hydrogen"
    
    # 4. 플렉스 연료
    elif "flex fuel" in eng or "e85" in eng:
        return "flex_fuel"
    
    # 5. 디젤
    elif "diesel" in eng or any(x in eng for x in ["ddi", "tdi", "crdi", "dci", "cdi", "hdi"]):
        return "diesel"
    
    # 6. 가솔린
    elif any(x in eng for x in [
        "gasoline", "gdi", "pdi", "v6", "v8", "liter", "turbo", "i4", "i3", "i6"
    ]):
        return "gasoline"
    
    # 7. 분류 불가
    else:
        return np.nan
