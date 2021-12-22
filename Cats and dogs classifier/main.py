import PySimpleGUI as sg
import os
import io
from pathlib import Path
from PIL import Image
from model import Model

def main_win():
    layout = []

    layout += [[sg.Column([[sg.Image(key='-IMAGE-')]], justification='center')]]
    layout += [[sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key='-IN-')]]
    layout += [[sg.Button('Load Image'), sg.Text('', key='-TEXT-')]]
    

    return sg.Window('Main window', layout, finalize=True)


main_window = main_win()
classification_model = Model()

while True:
    main_event, main_values = main_window.Read()

    if main_event is None or main_event == 'Exit':
        break
    elif main_event == 'Load Image':
        filename = Path(main_values['-IN-'])

        if os.path.exists(filename):
            image = Image.open(filename)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format='PNG')
            main_window['-IMAGE-'].update(data=bio.getvalue())

            proba = classification_model.classify_image(filename)

            proba_proc_dog = int((proba[0][0])*100)
            proba_proc_cat = int((1-proba[0][0])*100)

            main_window['-TEXT-'].update(f'Dog: {proba_proc_dog}% Cat: {proba_proc_cat}%')




