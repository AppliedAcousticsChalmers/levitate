cl -c CyclicUltrahapticsArray.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include" || @echo Cannot compile CyclicUltrahapticsArray && exit /b 1
cl -c TCPArray.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include" || @echo Cannot compile TCPArray && exit /b 1
cl -c array_control.cpp /std:c++14 -I"C:\Program Files\Ultrahaptics\include" || @echo Cannot compile array_control && exit /b 1
link array_control.obj TCPArray.obj CyclicUltrahapticsArray.obj /LIBPATH:"C:\Program Files\Ultrahaptics\lib" Ultrahaptics.lib libusb-1.0.lib Ws2_32.lib /OUT:array_control.exe || @echo Cannot link executable && exit /b 1
del *.obj