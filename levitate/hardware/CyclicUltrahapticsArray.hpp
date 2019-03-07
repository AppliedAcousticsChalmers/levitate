#include <cstdlib>
#include <math.h>
#include <complex>
#include <vector>
#include <atomic>

#include <UltrahapticsStateEmitters.hpp>

#ifdef _WIN32
    #define OS_WINDOWS
#endif

using namespace std;

typedef Ultrahaptics::States::StreamingStateEmitter<Ultrahaptics::States::TransducersState> SEmitter;
typedef Ultrahaptics::States::StateOutputInterval<Ultrahaptics::States::TransducersState> SInterval;

#define PI 3.14159265

class CyclicUltrahapticsArray 
{
    Ultrahaptics::DriverLibrary driver_library;
    Ultrahaptics::DriverDevice* device = nullptr;
    SEmitter* emitter;

protected:
    atomic<bool> emitting = {0}; // Holds the output bool
    atomic<float> amplitude_factor = {0};
    atomic<int> current_state = {0}; // Keeps track of the current state to output

    int num_transducers, state_delay;
    vector< vector< complex<float> > > states;
    

public:
    static int verbose;  
    // 0: Default, no output as long as everyting works
    // 1: Minor info: Some info about the system and major retults
    // 2: Flow info: To follow and know the current state
    // 3: Debugging: Show a lot of output, echo commands etc.
    static bool no_array;
    CyclicUltrahapticsArray();
    ~CyclicUltrahapticsArray();
    virtual bool interact(string line);
    inline bool interact() {return interact(input());}
    virtual string help();

protected:
    virtual inline string input() {string line; getline(cin, line); return line;}
    virtual inline void output(string contents) {cout << contents << endl;}
    inline void verbose_output(int level, string contents) {if (verbose >= level) {cout << contents << endl;}}
    template<typename ... Args>
    inline string stringer(const Args& ... args) {ostringstream oss; int a[] = {0, ((void)(oss << args), 0) ... }; return oss.str();}
    int ultrahapticsConnect();
    int ultrahapticsStart();
    int ultrahapticsStop();
    int timerStart();
    int readStatesFromFile(const char filename[]);

private:
    static void emissionCallback(SEmitter& emitter, SInterval& interval, const Ultrahaptics::HostTimePoint& deadline, void* user_ptr);
    static void timer(int* state_delay, void* user_ptr);
};


class TCPArray : public CyclicUltrahapticsArray
{
    int sock;
    

public:
    TCPArray(const char ip_address[], const char port[]);
    bool interact(string line);
    inline bool interact() {return interact(input());}
    string input();
    void output(string contents);
    string help();
};

inline string CyclicUltrahapticsArray::help()
{   
    string str = 
    "All interaction are handled with string commands.\n"
    "Most commands take parameters, and optional parameters are indicated with hard brackets.\n"
    "Leaving an optional parameter out will most of the time query the current value.\n"
    "The command are listed below. All commands should be passed in lowercase, and the optional part of the command is indicated.\n"
    "List of input commands:\n\n"
    "help\n\tDisplay this message.\n"
    "emit[ting] [on|off]\n\tTurn the aray on or off.\n"
    "amp[litude] [float in range [0,1]]\n\tControl global amplitude scaling.\n"
    "quit | emit\n\tTerminate the program gracefully.\n"
    "next [int]\n\tGo to the next state. The optional parameter indicates to jump multiple states.\n"
    "prev[ious] [int]\n\tGo to the previous state. The optional parameter indicates to jump multiple states.\n"
    "rate [float]\n\tSet or get the state transition rate.\n"
    "ind[ex] [int]\n\tSet or get the current state index.\n"
    "printstates [int]\n\tPrints stored states. Leave optional index to print all states.\n"
    "trans[ducers] pos[itions]\n\tPrint the positions of all the transducers on the form (x, y, z) on separates lines.\n"
    "trans[ducers] norm[als]\n\tPrint the normals of all the transducers on the form (nx, ny, nz) on separates lines.:\n"
    "trans[ducers] count\n\tPrint the number of transducers.\n"
    "file filename\n\tSpecify a file from which to read states. The file is assumed to hold bytes with complex floats.\n";
    return str;
}

inline string TCPArray::help()
{
    string str = CyclicUltrahapticsArray::help() + "states int\n\tPrepare to receive a number of states over TCP. The states should be sent as bytes with complex floats.\n";
    return str;
}

