import serial
Ser = serial.Serial('/dev/ttyACM0', 9600)
import sys
PYTHON2 = sys.version_info.major == 2
while True:
    print("Enter 0 to grasp and 1 to release")
    if PYTHON2:
        input_data = raw_input()
    else:
        input_data = input()
    print(input_data)
    if input_data == '0':
        print("sending 1")
        Ser.write('1'.encode())
    elif input_data == '1':
        print("sending 0")
        Ser.write('0'.encode())