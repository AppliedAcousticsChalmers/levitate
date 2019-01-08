#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include <complex>
#include <vector>

#include <thread>
#include <UltrahapticsStateEmitters.hpp>
#include <UltrahapticsDriver.hpp>
#include <UltrahapticsDriverDevice.hpp>

using namespace std;

typedef Ultrahaptics::States::StreamingStateEmitter<Ultrahaptics::States::TransducersState> SEmitter;
typedef Ultrahaptics::States::StateOutputInterval<Ultrahaptics::States::TransducersState> SInterval;

#define PI 3.14159265

vector< vector< complex<float> > > states;
vector< complex<float> > single_state;

struct callback_data
{
    atomic<bool> emitting; // Holds the output bool
    atomic<int> current_state; // Keeps track of the current state to output
    atomic<float> amplitude_factor;
};

void emission_callback(SEmitter& emitter, SInterval& interval, const Ultrahaptics::HostTimePoint& deadline, void* user_ptr) {
    callback_data* options = static_cast<callback_data*>(user_ptr);
    for (auto it = interval.begin(); it != interval.end(); ++it) {
        if (options->emitting) {
            single_state = states[options->current_state];
            for (int trans=0; trans<single_state.size(); trans++) {
                it->state().complexActivationAt(trans) = single_state[trans] * (float)options->amplitude_factor;
            }
        } else {
            it->state().setZero();
        }
    }
}

void timer(int* state_delay, void* user_ptr) {
    while (true) {
        callback_data* options = static_cast<callback_data*>(user_ptr);
        if (*state_delay > 0) {
            options->current_state = (options->current_state + 1) % states.size();
            this_thread::sleep_for(chrono::milliseconds(*state_delay));
        } else{
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        cerr << "Usage: " << argv[0] <<  " <filename>" << endl;
        return 1;
    }

    ifstream f(argv[1], ios::in | ios::binary);

    if (!f) {
        cerr << "Could not open file '" << argv[1] << "'!" << endl;
        return 1;
    }

    //Create driver library 
    Ultrahaptics::DriverLibrary driver_library;
    Ultrahaptics::DriverDevice* device = nullptr;
    SEmitter* emitter;
    // Loop while trying to connect
    
    cout << "Attempting to connect to Ultrahaptics Device...\t";    
    while (!device){
        cout.flush();
        device = Ultrahaptics::DriverDevice::getDriverDeviceByCapability(driver_library, 
            Ultrahaptics::DeviceCapabilities::DEVICE_CAPABILITY_TRANSDUCER_STATE_WITH_TIME_POINT);
        if (!device || !device->isConnected()){
            cout << "Not connected...\t";
            cout.flush();
            if (device) {
                Ultrahaptics::DriverDevice::destroyDevice(device);
            }
            device = nullptr;
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }
    cout << "Connection successful!" << endl;

    const char* firmware_version = device->getFirmwareVersion();
    cout << "Firmware version: " << firmware_version << endl;

    const int num_transducers = device->getTransducers().size();
    int read_count = 0;
    
    complex<float> state_read_buffer[num_transducers];
    // vector< complex<float> > single_state;
    // vector< typeof(single_state) > states;

    while (f.read((char*) state_read_buffer, sizeof(state_read_buffer))){
        states.push_back(vector< complex<float> > (state_read_buffer, state_read_buffer + num_transducers));
        read_count++;
    }
    cout << "Read " << read_count << " states" << endl;

    f.close();

    callback_data options;
    options.emitting = false;
    options.current_state = 0;
    options.amplitude_factor = 0;
    emitter = new SEmitter(device);
    emitter->setEmissionCallback(emission_callback, &options);
    emitter->start();
    cout << "Current update rate is: " << emitter->getUpdateRate() << endl;

    int state_delay;
    state_delay = 0;
    thread timer_thread(timer, &state_delay, &options);
    timer_thread.detach();

    string command;
    while (true) {
        cout << "Enter 'on' or 'off' to toggle output, 'a' to set overall amplitude, "
        "'n' to go to the next state, 't <ms>' to specify state intervals, "
        "'g <state>' to go to a specific state, "
        "'i' to show current state index, p <s>' to print states, 'q' to exit. ";
        cin >> command;
        if (command == "on") {
            options.emitting = true;
        } else if (command == "off") {
            options.emitting = false;
        } else if (command == "a") {
            float amplitude;
            cin >> amplitude;
            if (cin.fail() || amplitude < 0 || amplitude > 1) {
                cout << "Not a valid amplitude, try again!" << endl;
                cin.clear();
            } else {
                options.amplitude_factor = amplitude;
            }
        } else if (command == "q") {
            options.emitting = false;
            break;
        } else if (command == "n") {
            options.current_state = (options.current_state + 1) % states.size();
        } else if (command == "t") {
            int time;
            cin >> time;
            if (cin.fail()) {
                cout << "Not a valid interval, try again!" << endl;
                cin.clear();
            } else {
                state_delay = time;
            }
        } else if (command == "g") {
            int state;
            cin >> state;
            if (cin.fail()) {
                cout << "Could not read input, try again!" << endl;
                cin.clear();
            } else if (state < 0 || state >= states.size()) {
                cout << "State not in range, must be >=0 and <" << states.size() << endl;
            } else {
                options.current_state = state;
            }
        } else if (command == "i") {
            cout << options.current_state << endl;
        } else if (command == "p"){
            int state;
            cin >> state;
            if (cin.fail()) {
                // Print all states
                cin.clear();
                for (state=0; state<states.size(); state++){
                    cout << "State " << state << endl;
                    for (int trans=0; trans<num_transducers; trans++) {
                        cout << "\tTransducer " << trans << ": " << states[state][trans] << endl;
                    }
                }
            } else if (state < states.size() && state >= 0){
                // Print the specified state
                cout << "State " << state << endl;
                for (int trans=0; trans<num_transducers; trans++) {
                    cout << "\tTransducer " << trans << ": " << states[state][trans] << endl;
                }
            } else {
                cout << "State " << state << " is not available. " << states.size() << " have been read from file" << endl;
            }
        } else {
            cout << "Cannot understand input!" << endl;
            cin.clear();
        }
        cin.ignore(numeric_limits<streamsize>::max(),'\n');
    }

    emitter->stop();
    delete emitter;
    emitter = nullptr;

    Ultrahaptics::DriverDevice::destroyDevice(device);
    device = nullptr;

    return 0;
}