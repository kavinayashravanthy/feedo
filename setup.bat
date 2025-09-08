@echo off
echo ================================
echo FASTAPI PROJECT SETUP STARTED
echo ================================

REM Step 1: Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Step 2: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Step 3: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Step 4: Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Installing manually...
    pip install fastapi uvicorn pandas transformers torch textblob nltk scikit-learn
)

REM Step 5: Run the FastAPI app
echo Starting FastAPI server...
uvicorn testpeice1:app --reload

pause
