#include <cstdlib>
#include <iostream>
// #include <fstream>
// #include <string>
// #include <sstream>
// #include <thread>

#include "CyclicUltrahapticsArray.hpp"

#ifdef OS_WINDOWS
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <sys/socket.h>
    #include <netdb.h>
#endif

using namespace std;


TCPArray::TCPArray(const char ip_address[], const char port[])
{
    verbose_output(1, "Attempting TCP connection...");

    #ifdef OS_WINDOWS
        WSADATA wsaData;
        int iResult;
        iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
        if (iResult != 0) {printf("WSAStartup failed: %d\n", iResult);}
    #endif

    int status;
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // AF_INET or AF_INET6 to force version
    hints.ai_socktype = SOCK_STREAM;

    if ((status = getaddrinfo(ip_address, port, &hints, &res)) != 0) {cerr <<  "getaddrinfo: " << gai_strerror(status) << endl;}
    if ((sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) < 0) {perror("socket");}
    if (connect(sock, res->ai_addr, res->ai_addrlen) == -1) {perror("connect");}
    freeaddrinfo(res);
    verbose_output(1, "TCP connection successful.");
}

string TCPArray::input()
{
    verbose_output(2, "Waiting for data over TCP");
    uint32_t message_len=0, received=0;
    int receive_len = 0;
    receive_len = recv(sock, (char*) &message_len, 4, 0);
    if (receive_len == 0) {return "quit";}
    verbose_output(2, stringer("Reading ", message_len, " bytes over TCP."));

    // char buf[message_len];
    char* buf = new char[message_len];
    while (message_len > 0) {
        receive_len = recv(sock, buf + received, message_len, 0);
        received += receive_len;
        message_len -= receive_len;
    }
    verbose_output(3, stringer("Read message: \"", string(buf), "\" from TCP"));
    return string(buf);
}

void TCPArray::output(string contents)
{
    uint32_t message_len = contents.size();
    verbose_output(2, stringer("Sending ", message_len, " bytes over TCP."));
    
    send(sock, (char*) &message_len, 4, 0);
    send(sock, contents.c_str(), contents.size(), 0);
}

bool TCPArray::interact(string line)
{
    stringstream input_stream(line);
    string command;
    input_stream >> command ;
    if (command.rfind("states") == 0)
    {
        int num_states;
        input_stream >> num_states;
        if (input_stream.fail()) {num_states = 1;}
        // complex<float> state_read_buffer[num_transducers];
        complex<float>* state_read_buffer = new complex<float>[num_transducers];
        current_state = 0;
        states.clear();
        for (int idx=0; idx<num_states; idx++) {
            // state_read_buffer = (complex<float>*) receive();
            verbose_output(2, stringer("Waiting for state ", idx, " over TCP"));
            uint32_t message_len = 0, received=0;
            int receive_len = 0;
            receive_len = recv(sock, (char*) &message_len, 4, 0);
            verbose_output(2, stringer("Reading ", message_len, " bytes over TCP."));
            while (message_len > 0) {
                receive_len = recv(sock, (char*) state_read_buffer + received, message_len, 0);
                received += receive_len;
                message_len -= receive_len;
            }
            verbose_output(3, stringer("Received state ", idx, " over TCP"));
            states.push_back(vector< complex<float> > (state_read_buffer, state_read_buffer + num_transducers));
        }

        return true;
    }
    else
    {
        return CyclicUltrahapticsArray::interact(line);
    }
}