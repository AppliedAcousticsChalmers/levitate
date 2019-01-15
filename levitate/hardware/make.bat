cl -c CyclicUltrahapticsArray.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include"
cl -c TCPArray.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include"
cl -c array_control.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include"
link array_control.obj TCPArray.obj CyclicUltrahapticsArray.obj /LIBPATH:"C:\Program Files\Ultrahaptics\lib" Ultrahaptics.lib libusb-1.0.lib Ws2_32.lib /OUT:array_control.exe
del *.obj