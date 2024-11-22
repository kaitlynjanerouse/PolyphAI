# PolyphAI

This is a machine learning tool designed to aid musicians in composing sheet music. Musicians can input a series of starter notes as the melody, and the tool will suggest complimentary notes in SATB (Soprano, Alto, Tenor, Bass) format. If time permits, the tool will also allow users to prompt the model with different genres and instruments, enabling musicians to further tailor the tool to their specific needs.

## Steps to Running the Model:
1. Create a virtual environment: _python3 -m venv my_env_

2. Activate the virtual environment:
   - For mac users, run _source my_env/bin/activate_
   - For windows users, run _my_env\Scripts\activate_

3. Install requirements: _pip install -r requirements.txt_

4. Kickoff model: _python3 polyphai.py_
   - Results will be stored in Code/Results
   - .xml files can be viewed in MuseScore

5. Deactivate the virtual environemnt: _deactivate_


For changes to the repository:
1. git add .
2. git commit -m "model result"
3. git push
