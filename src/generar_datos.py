import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_patient_data(n=1000):
    
    #Generates randomised patient data for n patients

    initial_date = datetime(2025,6,1)    
    data = []
    reasons = {'anxiety' : 0.4, 'depression' : 0.3, 'stress' : 0.15, 'relationship conflict' : 0.15}


    for i in range(n):

        patient_id = f'P{i+1:04}'
        age = np.random.randint(18, 75)
        gender = str(np.random.choice(['M', 'F'], p=[0.45, 0.55]))

        therapy_type = str(np.random.choice(['individual', 'couple', 'family'], p=[0.6, 0.3, 0.1]))
        therapist = str(np.random.choice(['Dra. Gutierrez', 'Dr. Dominguez', 'Dra. Rodriguez', 'Dr. Min']))
        reason = str(np.random.choice(list(reasons.keys()), p = list(reasons.values())))

        sessions_number = np.random.randint(1, 21)
        first_session_date = initial_date - timedelta(np.random.randint(30, 365))
        frecuency = int(np.random.choice([7, 14], p=[0.7, 0.3])) #weekly or biweekly
        last_session_date = first_session_date + timedelta(days=(sessions_number-1)*frecuency)
    
        #Check for abandonment
        days_since_last = (datetime(2026,1,1) - last_session_date).days
        abandonment = 1 if days_since_last > 60 else 0

        #Risk factor for abandonment by age
        if age < 30:
            additional_abandonment_probablity = 0.15
        elif age > 60:
            additional_abandonment_probablity = 0.1
        else:
            additional_abandonment_probablity = 0
        
        if therapy_type == 'couple':
            additional_abandonment_probablity += 0.2
        
        if np.random.random() < additional_abandonment_probablity:
            abandonment = 1

        #Low compromise

        if sessions_number< 3 and abandonment == 1:
            low_compromise = 1
        else:
            low_compromise = np.random.choice([0, 1], p=[0.7, 0.3])

        data.append({
            'patient_id' : patient_id,
            'age' : age,
            'gender' : gender,
            'therapy_type' : therapy_type,
            'therapist' : therapist,
            'reason' : reason,
            'sessions_number' : sessions_number,
            'first_session_date' : first_session_date,
            'last_session_date' : last_session_date,
            'days_since_last' : days_since_last,
            'low_compromise' : low_compromise,
            'abandonment' : abandonment
        })


    return pd.DataFrame(data)