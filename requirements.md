when installing 

`navigate to hcp-engagement-api/`

create a venv first: 

```
python3 -m venv venv
```

start the venv: 

```
source venv/bin/activate 
```

once youre inside of the venv, being installing the packages: 

check your python version: 

```
python3 --version
```

```
pip install --upgrade pip
pip install -r requirements.txt
```

if you had errors, it is likely you are on a newer version of python than what was used to create this api

if youre on python 3.11.9, this requirements.txt file should install everything properly 
if youre on python >= 3.12, you will likely have errors when installing the requirements, follow these steps: 

begin running the app to see what dependencies were missing during install: 

```
python3 app.py
```

it will say ModuleNotFoundError: No Module named '{package}'

for any packages that show as not yet installed, just run:

pip install {package}

likely packages that were not preinstalled: 
- flask
- flask_restx
- flask_cors
- flask_socketio
- passlib
- requests
- dotenv
- pandas
- bcrypt

continue to repeat these steps running `python3 app.py`

for some packages, they will have version mismatching, so you need to install the exact version that was listed in requirements.txt

this likely includes: 
- bcrypt
- jwt

for these scripts, specifically use the version specified in requirements.txt: 

for example: 

```
pip install bcrypt==4.0.1
```
