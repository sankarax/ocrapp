This is the official repo of the OCR web app designed for the MLBD course. 
To get the web app working, a few steps need to be followed. 
- Install node.js from  https://nodejs.org/en/download  (to run the frontend). Add to PATH variable if necessary
- Download the ocrapp repo as a zip file
- Once inside the folder, create a virtual environment if needed (optional)
- Open the terminal from this folder in any editor (VS Code is tried and tested)
- Run this command "pip install -r requirements.txt"      (This installs all the dependecies for the backend automatically)
- Now move to the client subfolder using "cd client"
- As we have node installed just run "npm install"  (This installs all dependencies for the web app)
- In case you get an error "..npm.ps1 cannot be loaded because running scripts is disabled on this system..." then run this command "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
- Everything is set, in order to run the backend, simply run the "OCRapp.py" from the project folder using the python interpreter
  The first run will take a while as the dataset needs to be downloaded. 
- The terminal will have some lines like this "Debugger is active! 
  Debugger PIN: 417-664-050"
- Now open a new terminal within the editor and go to the client folder "cd client". If running in a virtual environment make sure it's activated again
- Run "npm run dev"   (This starts the frontend).  Make sure the project path doesn't have weird spaces or special characters as there was an issue that arised
- Navigate to the address shown there (http://localhost:5173/) for example
- Now upload any of the images available in the image folder or simply upload a new image with your own text
  to get the prediction. Voila!

A few more points. The CNN (EMNIST), resnet and vit notebook files are available in the repo as well for testing. Simply execute the cells
in the files in order to train the model for example. But by default the trained models are already located in the "models" folder 
so simply execute the cells which load the parameters to the initialized models. 
