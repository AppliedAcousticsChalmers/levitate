#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>

#include "CyclicUltrahapticsArray.hpp"

using namespace std;

int CyclicUltrahapticsArray::verbose = 0;
bool CyclicUltrahapticsArray::no_array = false;

CyclicUltrahapticsArray::CyclicUltrahapticsArray()
{
    verbose_output(1, stringer("Verbosity level is: ", verbose));
    if (ultrahapticsConnect() != 0) {cerr << "Could not connect to ultrahaptics array!\n";}
    if (ultrahapticsStart() != 0) {cerr << "Could not start array callback loop\n";}
    if (timerStart() != 0) {cerr << "Could not start timer thread\n";}
};

CyclicUltrahapticsArray::~CyclicUltrahapticsArray()
{
    if (ultrahapticsStop() != 0) {cerr << "Could not halt array properly!\n";}
}

int CyclicUltrahapticsArray::ultrahapticsConnect()
{
    verbose_output(1, "Attempting to connect to Ultrahaptics device...");
    if (no_array) {num_transducers = 4; return 1;}  // Use 4 transducer for debugging.
    while (!device){
        device = Ultrahaptics::DriverDevice::getDriverDeviceByCapability(driver_library, 
            Ultrahaptics::DeviceCapabilities::DEVICE_CAPABILITY_TRANSDUCER_STATE_WITH_TIME_POINT);
        if (!device || !device->isConnected()){
            verbose_output(3, "Not connected...");
            if (device) {
                Ultrahaptics::DriverDevice::destroyDevice(device);
            }
            device = nullptr;
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
    num_transducers = device->getTransducers().size();
    verbose_output(1, stringer("Connection successful, array uing ", num_transducers, " transducers."));
    return 0;
}

int CyclicUltrahapticsArray::ultrahapticsStart()
{
    verbose_output(1, "Starting Ultrahaptics array...");
    if (no_array) {return 1;}
    emitting = false;
    current_state = 0;
    amplitude_factor = 0;
    emitter = new SEmitter(device);
    emitter->setEmissionCallback(emissionCallback, this);
    emitter->start();
    verbose_output(1, stringer("Current update rate is: ", emitter->getUpdateRate()));
    verbose_output(1, stringer("Firmware version: ", device->getFirmwareVersion()));
    
    return 0;
}

int CyclicUltrahapticsArray::ultrahapticsStop()
{
    verbose_output(1, "Stopping Ultrahaptics array");
    if (no_array) {return 1;}
    emitter->stop();
    delete emitter;
    emitter = nullptr;
    Ultrahaptics::DriverDevice::destroyDevice(device);
    device = nullptr;
    return 0;
}

int CyclicUltrahapticsArray::timerStart()
{
    verbose_output(2, "Starting timer thread...");
    state_delay = 0;
    thread timer_thread(timer, &state_delay, this);
    timer_thread.detach();
    return 0;
}

int CyclicUltrahapticsArray::readStatesFromFile(const char filename[])
{
    verbose_output(2, stringer("Starting read from file: ", filename));
    ifstream f(filename, ios::in | ios::binary);

    if (!f) {cerr << "Could not open file '" << filename << "'!" << endl; return -1;}

    int read_count = 0;
    current_state = 0;
    complex<float>* state_read_buffer = new complex<float>[num_transducers];
    states.clear();
    while (f.read((char*) state_read_buffer, num_transducers * sizeof(state_read_buffer[0]))){
        states.push_back(vector< complex<float> > (state_read_buffer, state_read_buffer + num_transducers));
        read_count++;
    }

    verbose_output(1, stringer("Read ", read_count, " states from file ", filename ));
    
    f.close();
    return read_count;
}

void CyclicUltrahapticsArray::emissionCallback(SEmitter& emitter, SInterval& interval, const Ultrahaptics::HostTimePoint& deadline, void* user_ptr)
{
    CyclicUltrahapticsArray* array = static_cast<CyclicUltrahapticsArray*>(user_ptr);
    for (auto it = interval.begin(); it != interval.end(); ++it) {
        if (array->emitting) {
            vector< complex<float> > single_state = array->states[array->current_state];
            for (int trans=0; trans<single_state.size(); trans++) {
                it->state().complexActivationAt(trans) = single_state[trans] * (float)array->amplitude_factor;
            }
        } else {
            it->state().setZero();
        }
    }
}

void CyclicUltrahapticsArray::timer(int* state_delay, void* user_ptr)
{
    while (true) {
        CyclicUltrahapticsArray* array = static_cast<CyclicUltrahapticsArray*>(user_ptr);
        if (*state_delay > 0 && array->states.size()>0) {
            array->current_state = (array->current_state + 1) % array->states.size();
            this_thread::sleep_for(chrono::microseconds(*state_delay));
        } else {
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
}

bool CyclicUltrahapticsArray::interact(string line)
{
    stringstream input_stream(line);
    string command;
    input_stream >> command ;
    verbose_output(3, stringer("Interact got command: ", command));

    if (command.rfind("help") == 0)
    {
        output(help());
    }
    else if (command.rfind("emit") == 0) 
    {
        input_stream >> command;
        if (input_stream.fail()) {
            output(emitting ? "on" : "off");
        } else if (command == "on") {
            emitting = true;    
        } else if (command == "off") {
            emitting = false;
        } else {
            cerr << "Unknown option `" << command << "`!\n";
        }
    } 
    else if (command.rfind("amp") == 0) 
    {
        float ampl;
        input_stream >> ampl;
        if (input_stream.fail()) {
            output(stringer(amplitude_factor));
        } else if (ampl < 0 || ampl > 1){
            cerr << "`" << ampl << "` is not a valid amplitude, try again!" << endl;
        } else {
            amplitude_factor = ampl;
        }
    } 
    else if (command.rfind("quit") == 0 || command.rfind("exit") == 0) 
    {
        emitting = false;
        return false;
    } 
    else if (command.rfind("next") == 0) 
    {
        int num;
        input_stream >> num;
        if (input_stream.fail()) { num = 1; }
        current_state = (current_state + num) % states.size();
    } 
    else if (command.rfind("prev") == 0) 
    {
        int num;
        input_stream >> num;
        if (input_stream.fail()) { num = 1; }
        current_state = (current_state - num) % states.size();
    } 
    else if (command.rfind("rate") == 0) 
    {
        float rate;
        input_stream >> rate;
        if (input_stream.fail()) {
            output(stringer(1000000 / state_delay));
        } else {
            state_delay = round(1000000 / rate);
            // output(2, stringer("Microsecond delay is: ", state_delay));
        }
    } 
    else if (command.rfind("ind") == 0) 
    {
        int state;
        input_stream >> state;
        if (input_stream.fail()) {
            output(stringer(current_state));
        } else if (state < 0 || state >= states.size()) {
            cerr << "State not in range, must be >=0 and <" << states.size() << endl;
        } else {
            current_state = state;
        }
    } 
    else if (command.rfind("printstates") == 0)
    {
        int state;
        input_stream >> state;
        if (input_stream.fail()) {
            // Print all states
            output(stringer("Displaying all ", states.size(), " states."));
            for (state=0; state<states.size(); state++){
                output(stringer("State ", state));
                for (int trans=0; trans<num_transducers; trans++) {
                    output(stringer("\tTransducer ", trans, ": ", states[state][trans]));
                }
            }
        } else if (state < states.size() && state >= 0){
            // Print the specified state
            output(stringer("State ", state));
            for (int trans=0; trans<num_transducers; trans++) {
                output(stringer("\tTransducer " , trans,  ": ", states[state][trans]));
            }
        } else {
            output(stringer("State ", state, " is not available. " , states.size(), " have been read."));
        }
    }
    else if (command.rfind("trans") == 0)
    {
        
        input_stream >> command;
        if (command.rfind("pos") == 0) {
            const Ultrahaptics::TransducerContainer& transducers = device->getTransducers();
            for (int i=0; i<transducers.size(); i++) {
                output(stringer("(", transducers[i].x_position, ", ", transducers[i].y_position, ", ", transducers[i].z_position, ")"));
            }
        } else if (command.rfind("norm") == 0) {
            const Ultrahaptics::TransducerContainer& transducers = device->getTransducers();
            for (int i=0; i<transducers.size(); i++) {
                output(stringer("(", transducers[i].x_upvector, ", ", transducers[i].y_upvector, ", ", transducers[i].z_upvector, ")"));
            }
        } else if (command.rfind("num") == 0 || command.rfind("count") == 0) {
            output(stringer(num_transducers));
        }
    } 
    else if (command.rfind("file") == 0)
    {
        input_stream >> command;
        readStatesFromFile(command.c_str());
    }
    else 
    {
        verbose_output(0, stringer("Cannot understand input `", command, "`. Type help for a list of commands!"));
    }
    return true;
}

