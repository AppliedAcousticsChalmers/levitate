#include <cstdlib>
#include <iostream>
#include <string>

#include "CyclicUltrahapticsArray.hpp"

CyclicUltrahapticsArray* parse_args(int argc, char *argv[]) {
    string filename;
    string ip;
    string port;
    for (int idx=1; idx<argc; idx++)
    {
        if (string(argv[idx]).rfind("-noarr") == 0) 
        {
            CyclicUltrahapticsArray::no_array = true;        
        }
        else if (string(argv[idx]).rfind("-v") == 0) 
        {
            CyclicUltrahapticsArray::verbose = string(argv[idx]).size()-1;
        }
        else if (string(argv[idx]).rfind("--verb") == 0)
        {
            CyclicUltrahapticsArray::verbose = stoi(argv[++idx]);
        }
        else if (string(argv[idx]).rfind("--ip") == 0)
        {
            ip = argv[++idx];
            port = argv[++idx];
        }
        else if (string(argv[idx]).rfind("--file") == 0)
        {
            filename = argv[++idx];
        }
    }

    if (ip.size() > 0) 
    {
        CyclicUltrahapticsArray* device = new TCPArray(ip.c_str(), port.c_str());
        if (filename.size() > 0) 
        {
            device->interact("filename " + filename);
        }
        return device;
    }
    else
    {
        CyclicUltrahapticsArray* device = new CyclicUltrahapticsArray;
        if (filename.size() > 0) 
        {
            device->interact("filename " + filename);
        }
        return device;
    }
}

int main(int argc, char *argv[]){
    
    

    // TCPArray device(argv[1], argv[2]);
    CyclicUltrahapticsArray* device = parse_args(argc, argv);

    
    bool running = true;
    while (running) {
        // string command;
        // cout << "Enter 'on' or 'off' to toggle output, 'a' to set overall amplitude, "
        //     "'n' to go to the next state, 't <ms>' to specify state intervals, "
        //     "'g <state>' to go to a specific state, "
        //     "'i' to show current state index, p <s>' to print states, 'q' to exit. ";
        // getline(cin, command);
        running = device->interact();
        // msg = device.input();
        // cout << msg << endl;
        // device.output(msg);
    }

    

    return 0;
}