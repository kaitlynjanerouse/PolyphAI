# PolyphAI

This is a machine learning tool designed to aid musicians in composing sheet music. Musicians can input a series of starter notes as the melody, and the tool will suggest complimentary notes in SATB (Soprano, Alto, Tenor, Bass) format. If time permits, the tool will also allow users to prompt the model with different genres and instruments, enabling musicians to further tailor the tool to their specific needs.

## Steps to Running the Model:
1. Create a virtual environment: 'python3 -m venv my_env'

2. Activate the virtual environment:
   - For mac users, run source my_env/bin/activate
   - For windows users, run my_env\Scripts\activate

3. Install requirements: pip install -r requirements.txt

4. Kickoff model: python3 polyphai.py
   Results will be stored in Code/Results
   .xml files can be viewed in MuseScore

5. Deactivate the virtual environemnt: deactivate
