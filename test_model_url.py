import requests


MAXIMO_VISUAL_INSPECTION_API_URL = 'https://mas83.visualinspection.maximo26.innovationcloud.info/api/dlapis/9ffb662b-790a-4fb2-a419-217fdf1ac0ce'

with open('test.jpg', 'rb') as f:
        # WARNING! verify=False is here to allow an untrusted cert!
        r = requests.post(MAXIMO_VISUAL_INSPECTION_API_URL,
                   files={'files': ('test.jpg', f)},
                   verify=False)
        print(r)