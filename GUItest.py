from asyncio import subprocess
from turtle import title
import PySimpleGUI as sg
from subprocess import run

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  
            [sg.Text('Posture Detector', size=(50, 10), font=('Any 30'),justification='center', text_color='White')],
            [sg.Button('Start Tracking', size=(20,1), font=('Any 20'))],
            [sg.Button('Exit', size=(20,1), font=('Any 20'))] 
            
        ]

# Create the Window
window = sg.Window('PostureTracker', layout, margins=(700, 600), element_justification='c')
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == "Start Tracking":
        run(['python3 visualizer.py --back Marthese_Trim_Flip.mp4 --side MartheseSide_Trim.mp4'], shell=True)
    elif event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    print('You entered ', values[0])

window.close()















# layout = [
#     [sg.Text('File 1'), sg.InputText(), sg.FileBrowse(),
#      sg.Checkbox('MD5'), sg.Checkbox('SHA1')
#      ],
#     [sg.Text('File 2'), sg.InputText(), sg.FileBrowse(),
#      sg.Checkbox('SHA256')
#      ],
#     [sg.Output(size=(88, 20))],
#     [sg.Submit(), sg.Cancel()]
# ]
# window = sg.Window('File Compare', layout)
# while True:                             # The Event Loop
#     event, values = window.read()
#     # print(event, values) #debug
#     if event in (None, 'Exit', 'Cancel'):
#         break