
def health_calculator():

    age=int(input('Enter your age: '))
    height=int(input('Enter your height in cm: '))
    weight=int(input('Enter your weight in kg: '))
    activity_level=input('Enter your activity level (sedentary = 1, lightly active =2, moderately active =3, very active= 4): ').lower()
    gender=input('Enter your gender please: ').lower()

    bmi=weight/((height/100)**2)
    
    if gender =='male':
        bmr=66.5 + (13.75 * weight) + (5.003 * height) - (6.775 * age)
        if activity_level == '1':
            if bmi< 18.5:
                print('Your BMI is below normal and you are sedentary. Consider inceasing your activity level and consult a doctor for healthy weight gain .')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are sedentary. Consider doing some light exercises to maintain your health.')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are sedentary. Consider increasing your activity level and consult a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are sedentary. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '2':
            if bmi < 18.5:
                print('Your BMI is below normal and you are lightly active. Consider increasing your activity level and consult a doctor for healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are lightly active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are lightly active. Consider increasing your activity level and consult a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are lightly active. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '3':
            if bmi < 18.5:
                print('Your BMI is below normal and you are moderately active. Consider consulting a doctor for some healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are moderately active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are moderately active. Consider consulting a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are moderately active. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '4': 
            if bmi < 18.5:
                print('Your BMI is below normal and you are very active. Consider consulting a doctor for some healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are very active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are very active. Consider consulting a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are very active. It is important to consult a doctor for a personalized health plan.')
        print(f'Your BMI is {bmi:.2f} and your BMR is {bmr:.2f} calories/day.')
    
    if gender == 'female':
        bmr=655.1 + (9.563 * weight) + (1.850 * height) - (4.676 * age)
        if activity_level=='1':
            if bmi < 18.5:
                print('Your BMI is below normal and you are sedentary. Consider increasing your activity level and consult a doctor for healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are sedentary. Consider doing some light exercises to maintain your health.')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are sedentary. Consider increasing your activity level and consult a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are sedentary. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '2':
            if bmi < 18.5:
                print('Your BMI is below normal and you are lightly active. Consider increasing your activity level and consult a doctor for healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are lightly active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are lightly active. Consider increasing your activity level and consult a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are lightly active. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '3':
            if bmi < 18.5:
                print('Your BMI is below normal and you are moderately active. Consider consulting a doctor for some healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are moderately active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are moderately active. Consider consulting a doctor for healthy weight loss.')
            elif bmi >= 30:
                print('Your BMI is obese and you are moderately active. It is important to consult a doctor for a personalized health plan.')
        elif activity_level == '4':
            if bmi < 18.5:
                print('Your BMI is below normal and you are very active. Consider consulting a doctor for some healthy weight gain.')
            elif 18.5 <= bmi < 24.9:
                print('Your BMI is normal and you are very active. Keep up the good work with your health!')
            elif 25 <= bmi < 29.9:
                print('Your BMI is above normal and you are very active. Consider consulting a doctor for healthy weight loss.')
            elif bmi >= 30:
                 print('Your BMI is obese and you are very active. It is important to consult a doctor for a personalized health plan.')

        print(f'Your BMI is {bmi:.2f} and your BMR is {bmr:.2f} calories/day.')
  
    
health_calculator()

    