/*
Projecte de Xarxa neuronal.
 - Perceptró multicapa dinàmic

 GuillemBM

 ORGANITZACIÓ:
	- Includes
	- Struct/typedefs
	- Function declarations
	- Global params
	- Main
	- Function definitions
*/

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
// INCLUDES //

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <random>

using namespace std;

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
// DEFINITIONS/TYPEDEFS //

struct Wire { 
        int layer, numI, numO;
        double omega;
    }; //nexus-nexus connections

typedef vector<Wire> WireVector;
typedef vector<WireVector> WireMatrix;
typedef vector<WireMatrix> WireWeb;
    
struct Nexus {
        bool p;
        int layer, num;
        double alpha, u;
    }; //nexus

typedef vector<Nexus> NexusLayer;
typedef vector<NexusLayer> NexusWeb;

typedef vector<double> DataVector;
typedef vector<DataVector> DataMatrix;
typedef vector<DataMatrix> DataWires;

class ThrowError{
    public:
    ThrowError(string err)
    {
        errText = err;
    }
    string getText()
    {
        return errText;
    }
    private:
    string errText;
};

std::random_device seed;
std::mt19937 rnd(seed());
std::uniform_real_distribution<> d(0, 1);

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
//FUNCTION DECLARATIONS

double rand_init(); //returns a rand double 0:1
void init_vars(const NexusWeb& nexusWeb, const WireWeb& wireWeb, DataMatrix& us, DataWires& omegas);

void fill(NexusWeb& nexusWeb, WireWeb& wireWeb, const int numInputs, const vector<int> hideLayers, const int numOutputs); //inits the webs with the params values

void neuron(Nexus& current, const NexusWeb& nexusWeb, const WireWeb& wireWeb); //neuron function (sum)
double sigmoid(const double alpha); //sigmoid function

void activate(NexusWeb& nexusWeb, const WireWeb& wireWeb, const DataVector& inputs); //update alpha values
void update_us(NexusWeb& nexusWeb, const DataMatrix& us);
void update_omegas(WireWeb& wireWeb, const DataWires& omegas);

void learn(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const DataVector& expectedOutputs, DataMatrix& us, DataWires& omegas, const double learnConst); //learn function
double derivate_u(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const Nexus& to, const Nexus& from);
double derivate_omega(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const Nexus& to, const Wire& from, int pas);
double neuron_derivate(const Nexus& neuron);

void print_outputs(const NexusWeb& nexusWeb, const DataVector& expectedOutputs); //prints the outputs
void print_info(const NexusWeb& nexusWeb, const WireWeb& wireWeb); //prints info

void test_inputs(NexusWeb& nexusWeb, const WireWeb& wireWeb); //test user inputs

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
// PARAMS [GLOBAL] //

const bool outParams = true;
const bool outData = true;

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
// MAIN //

int main()
{
    NexusWeb nexusWeb;
    WireWeb wireWeb;

    DataMatrix us;
    DataWires omegas;

    //-------------------------------------------------------
    // PARAMS [LOCAL] //

    int numInputs = 2; //num inputs
    int numOutputs = 1; //num outputs
    vector<int> hideLayers = {}; //hidden layers
    double learnConst = 0.4; //learn const (alph)
    double aproxDelta = 0.1; //aprox error

    int numCases = 4; //num cases to teach

    //-------------------------------------------------------
    // PARAMS //
    
    
    ifstream inputParams;
    inputParams.open("params.txt");

    if(outParams)
    {
        inputParams >> numInputs;

        inputParams >> numOutputs;

        for(int tmp; inputParams >> tmp and tmp != -1; ) hideLayers.push_back(tmp);

        inputParams >> learnConst;

        inputParams >> aproxDelta;

        inputParams >> numCases;
    }

    cout << "-------------------------------------------" << endl;
    cout << " # WEB INFO #" << endl;
    cout << "--------------" << endl;
    cout << "Num. Inputs: " << numInputs << endl;
    cout << "Num. Outputs: " << numOutputs << endl;
    cout << "Num. Hiden Layers: " << hideLayers.size() << " -> ";
    cout << " [ ";
    for(unsigned int i = 0; i < hideLayers.size(); ++i) cout << hideLayers[i] << " " ;
    cout << "]" << endl << endl;
    cout << "Learn Const.: " << learnConst << endl;
    cout << "Aprox. delta: " << aproxDelta << endl << endl;
    cout << "Cases to teach: " << numCases << endl;
    cout << "-------------------------------------------" << endl;



    //-------------------------------------------------------
    // IN/OUT //
    
    DataMatrix inputs(numCases, DataVector(numInputs));
    DataMatrix expectedOutputs(numCases, DataVector(numOutputs));

    ifstream inputData;
    inputData.open("inputs.txt");
    ifstream outputData;
    outputData.open("outputs.txt");

    if(outData)
    {
        for(int i = 0; i < numCases; ++i)
        {
            for(int j = 0; j < numInputs; ++j)
            {
                inputData >> inputs[i][j];
            }
        }   
    }
    else
    {
        inputs = {
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1}
        };
    }

    if(outData)
    {
        for(int i = 0; i < numCases; ++i)
        {
            for(int j = 0; j < numOutputs; ++j)
            {
                outputData >> expectedOutputs[i][j];
            }
        }
    }
    else
    {
        expectedOutputs = {
            {0, 0},
            {0, 1},
            {0, 1},
            {1, 0},
            {0, 1},
            {1, 0},
            {1, 0},
            {1, 1}
        };
    }
    
    //-------------------------------------------------------
    // TEST BLOCK //

    try
    {
        fill(nexusWeb, wireWeb, numInputs, hideLayers, numOutputs);

        init_vars(nexusWeb, wireWeb, us, omegas);

        update_us(nexusWeb, us);
        update_omegas(wireWeb, omegas);

        print_info(nexusWeb, wireWeb);
    }
    catch(ThrowError& err)
    {
        cerr << "Error: " << err.getText() << endl;
    }

    //-------------------------------------------------------
	// LEARNING LOOP //
    
    bool learned = false;

    cout << "Learning..." << endl;

    while(not learned)
    {
        learned = true;

        for(int i = 0; i < numCases; ++i)
        {
            //cout << "------------------------------------------" << endl;
            //cout << "TESTING CASE # " << i+1 << " #" << endl;

            activate(nexusWeb, wireWeb, inputs[i]);

            //print_outputs(nexusWeb, expectedOutputs[i]);
            
            for(unsigned int j = 0; learned and j < nexusWeb[nexusWeb.size()-1].size(); ++j)
            {
                if((nexusWeb[nexusWeb.size()-1][j].alpha < expectedOutputs[i][j]-aproxDelta) or (nexusWeb[nexusWeb.size()-1][j].alpha > expectedOutputs[i][j]+aproxDelta))
                {
                    learned = false;
                }
            }

            if(not learned)
            {
                learn(nexusWeb, wireWeb, expectedOutputs[i], us, omegas, learnConst);
                update_us(nexusWeb, us);
                update_omegas(wireWeb, omegas);
            } 
        }
    }

    cout << "-------------------------------------------------------" << endl;
    cout << "                   ### LEARNED! ###" << endl;
    cout << "-------------------------------------------------------" << endl;
    
	//-------------------------------------------------------
    // RESULTS //
    
    cout << "-------------------------------------------------------" << endl;
    cout << "RESULTS:" << endl;

    for(int i = 0; i < numCases; ++i)
	{
        cout << "---------------------------------------" << endl;
        cout << "CASE: " << i+1 << endl;
        cout << "Inputs: [ ";
        for(unsigned int j = 0; j < inputs[i].size(); ++j) cout << "X" << j+1 << " = " << inputs[i][j] << ", ";
        cout << "]" << endl << endl;
        activate(nexusWeb, wireWeb, inputs[i]);
        print_outputs(nexusWeb, expectedOutputs[i]);
    }
    
    cout << "-------------------------------------------------------" << endl;

    //-------------------------------------------------------

    while(true) test_inputs(nexusWeb, wireWeb);

}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//
// FUNCTION DEFINITIONS

//-------------------------------------------------------

double rand_init()
{
    return d(rnd);
}

//-------------------------------------------------------

void init_vars(const NexusWeb& nexusWeb, const WireWeb& wireWeb, DataMatrix& us, DataWires& omegas)
{
    for(unsigned int i = 0; i < nexusWeb.size(); ++i)
    {
        DataVector tmpUL;
        for(unsigned int j = 0; j < nexusWeb[i].size(); ++j)
        {
            if(i == 0) tmpUL.push_back(0);
            else tmpUL.push_back(rand_init());
        }

        us.push_back(tmpUL);
    }

    for(unsigned int i = 0; i < wireWeb.size(); ++i)
    {
        DataMatrix tmpOM;
        for(unsigned int j = 0; j < wireWeb[i].size(); ++j)
        {
            DataVector tmpOV;
            for(unsigned int k = 0; k < wireWeb[i][j].size(); ++k)
            {
                tmpOV.push_back(rand_init());
            }
            tmpOM.push_back(tmpOV);
        }
        
        omegas.push_back(tmpOM);
    }

    cout << "INIT VARS > OK" << endl;
}

//-------------------------------------------------------

void fill(NexusWeb& nexusWeb, WireWeb& wireWeb, const int numInputs, const vector<int> hideLayers, const int numOutputs)
{
    if(numInputs == 0) throw ThrowError("0 inputs");
    if(numOutputs == 0) throw ThrowError("0 outputs");
    //NEXUSWEB INIT

    NexusLayer tmpIL;

    for(int i = 1; i <= numInputs; ++i)
    {
        Nexus tmpIN;

        tmpIN.p = true;
        tmpIN.layer = 1;
        tmpIN.num = i;

        tmpIN.u = 0;
        tmpIN.alpha = 0;

        tmpIL.push_back(tmpIN);
    }

    nexusWeb.push_back(tmpIL);

    for(unsigned int i = 1; i <= hideLayers.size(); ++i)
    {
        NexusLayer tmpHL;

        for(int j = 1; j <= hideLayers[i-1]; ++j)
        {
            Nexus tmpHN;

            tmpHN.p = false;
            tmpHN.layer = i+1;
            tmpHN.num = j;

            tmpHN.u = 1; //Not random?
            tmpHN.alpha = 0;

            tmpHL.push_back(tmpHN);
        }
        
        nexusWeb.push_back(tmpHL);
    }

    NexusLayer tmpOL;

    for(int i = 1; i <= numOutputs; ++i)
    {
        Nexus tmpON;

        tmpON.p = false;
        tmpON.layer = hideLayers.size()+2;
        tmpON.num = i;

        tmpON.u = 1; //Not random?
        tmpON.alpha = 0;

        tmpOL.push_back(tmpON);
    }

    nexusWeb.push_back(tmpOL);

    //WIREWEB INIT

    for(unsigned int i = 1; i <= nexusWeb.size()-1; ++i)
    {
        WireMatrix tmpWM;

        for(unsigned int j = 1; j <= nexusWeb[i-1].size(); ++j)
        {
            WireVector tmpWV;
            for(unsigned int k = 1; k <= nexusWeb[i].size(); ++k)
            {
                Wire tmpW;

                tmpW.layer = i;
                tmpW.numI = j;
                tmpW.numO = k;

                tmpW.omega = 1; //Not random?

                tmpWV.push_back(tmpW);
            }

            tmpWM.push_back(tmpWV);
        }

        wireWeb.push_back(tmpWM);
    }

    cout << "WEBS INIT > OK" << endl;
}

//-------------------------------------------------------

void neuron(Nexus& current, const NexusWeb& nexusWeb, const WireWeb& wireWeb) //NOT WORKS WITH FIRST LAYER (inputs)
{
    double alpha = 0;
    
    alpha += current.u;
    
    for(unsigned int i = 0; i < nexusWeb[current.layer-2].size(); ++i)
    {
        alpha += (nexusWeb[current.layer-2][i].alpha * wireWeb[current.layer-2][i][current.num-1].omega);
    }

    current.alpha = sigmoid(alpha);
}

//-------------------------------------------------------

double sigmoid(const double alpha)
{
	return 1 / (1 + exp(-alpha));
}

//-------------------------------------------------------

void activate(NexusWeb& nexusWeb, const WireWeb& wireWeb, const DataVector& inputs)
{
    for(unsigned int i = 0; i < nexusWeb[0].size(); ++i) //updates inputs
    {
        nexusWeb[0][i].alpha = inputs[i];
    }

    for(unsigned int i = 1; i < nexusWeb.size(); ++i)
    {
        for(unsigned int j = 0; j < nexusWeb[i].size(); ++j)
        {
            neuron(nexusWeb[i][j], nexusWeb, wireWeb);
        }
    }

}

//-------------------------------------------------------

void update_us(NexusWeb& nexusWeb, const DataMatrix& us)
{
    for(unsigned int i = 1; i < nexusWeb.size(); ++i)
    {
        for(unsigned int j = 0; j < nexusWeb[i].size(); ++j)
        {
            nexusWeb[i][j].u = us[i][j];
        }
    }
}

//-------------------------------------------------------

void update_omegas(WireWeb& wireWeb, const DataWires& omegas)
{
    for(unsigned int i = 0; i < wireWeb.size(); ++i)
    {
        for(unsigned int j = 0; j < wireWeb[i].size(); ++j)
        {
            for(unsigned int k = 0; k < wireWeb[i][j].size(); ++k)
            {
                wireWeb[i][j][k].omega = omegas[i][j][k];
            }
        }
    }
}

//-------------------------------------------------------

void learn(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const DataVector& expectedOutputs, DataMatrix& us, DataWires& omegas, const double learnConst)
{
    
    for(unsigned int i = 1; i < nexusWeb.size(); ++i)
    {
        for(unsigned int j = 0; j < nexusWeb[i].size(); ++j)
        {
            double error = 0;

            //INFO// cout << "--------------------------------------" << endl;
            //INFO// cout << "INICI U: " << i+1 << ", " << j+1 << endl;

            for(unsigned int o = 0; o < nexusWeb[nexusWeb.size()-1].size(); ++o)
            {
                //INFO// cout << "-- Per a sortida: " << o+1 << endl;
                error += ((nexusWeb[nexusWeb.size()-1][o].alpha - expectedOutputs[o]) * derivate_u(nexusWeb, wireWeb, nexusWeb[nexusWeb.size()-1][o], nexusWeb[i][j]));
            }

            us[i][j] -= learnConst * error;
        }
    }
    
    
    for(unsigned int i = 0; i < wireWeb.size(); ++i)
    {
        for(unsigned int j = 0; j < wireWeb[i].size(); ++j)
        {
            for(unsigned int k = 0; k < wireWeb[i][j].size(); ++k)
            {
                double error = 0;

                //INFO// cout << "--------------------------------------" << endl;
                //INFO// cout << "INICI OMEGA: " << i+1 << ", " << j+1 << ", " << k+1 << endl;

                for(unsigned int o = 0; o < nexusWeb[nexusWeb.size()-1].size(); ++o)
                {
                    //INFO// cout << "-- Per a sortida: " << o+1 << endl;
                    error += ((nexusWeb[nexusWeb.size()-1][o].alpha - expectedOutputs[o]) * derivate_omega(nexusWeb, wireWeb, nexusWeb[nexusWeb.size()-1][o], wireWeb[i][j][k], 0));
                }

                omegas[i][j][k] -= learnConst * error;
            }
        }
    }
    
}

//-------------------------------------------------------

double derivate_u(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const Nexus& to, const Nexus& from)
{
    if(from.layer == to.layer)
    {
        //INFO// cout << "Inici a la ultima layer. U: " << from.layer << ", " << from.num << "-> ( to" << to.num << ")" << endl;
        if(from.num != to.num) { /*cout << "NOP!" << endl;*/ return 0;}
        return neuron_derivate(to);
    }
    else
    {
        double sum_derivate = 0, init = 1;
        //INFO// cout << "Layer| Curr: " << from.layer << ", " << from.num << endl;
        init = neuron_derivate(nexusWeb[from.layer-1][from.num-1]);

        for(unsigned int i = 0; i < wireWeb[from.layer-1][from.num-1].size(); ++i)
        {
            //INFO// cout << "Cami cap a omega: " << from.layer << ", " << from.num << ", " << i+1 << "-> derivate nexus: " << from.layer+1 << ", " << i+1 << endl;
            sum_derivate += (wireWeb[from.layer-1][from.num-1][i].omega * derivate_u(nexusWeb, wireWeb, to, nexusWeb[from.layer][i]));
        }
        
        return (init * sum_derivate);
    }
}

//-------------------------------------------------------

double derivate_omega(const NexusWeb& nexusWeb, const WireWeb& wireWeb, const Nexus& to, const Wire& from, int pas)
{
    if(pas == 0 and from.layer+1 == to.layer)
    {
        //INFO// cout << "Inici a la ultima layer. Omega: " << from.layer << ", " << from.numI << ", " << from.numO << " -> ( to" << to.num << ")" << endl;
        if(from.numO != to.num) { /*cout << "NOP!" << endl;*/ return 0; }
        return (nexusWeb[from.layer-1][from.numI-1].alpha * neuron_derivate(to));
    }
    else if(pas != 0 and from.layer+1 == to.layer)
    {
        //INFO// cout << "Ultima layer. Omega: " << from.layer << ", " << from.numI << ", " << from.numO << " -> ( to" << to.num << ")" << endl;
        if(from.numO != to.num) { /*cout << "NOP!" << endl;*/ return 0; }
        return (from.omega * neuron_derivate(to));
    }
    else
    {
        double sum_derivate = 0, init = 1;

        if(pas == 0) 
        {
            //INFO// cout << "Primera layer| Pre: " << from.layer << ", " << from.numI << ", Post: " << from.layer+1 << ", " << from.numO << endl;
            init = (nexusWeb[from.layer-1][from.numI-1].alpha * neuron_derivate(nexusWeb[from.layer][from.numO-1]));
        }
        else 
        {
            //INFO// cout << "Mitg| Omega: " << from.layer << ", " << from.numI << ", " << from.numO << " -> NEXUS: " << from.layer+1 << ", " << from.numO << endl;
            init = (from.omega * neuron_derivate(nexusWeb[from.layer][from.numO-1]));
        }

        for(unsigned int i = 0; i < wireWeb[from.layer][from.numO-1].size(); ++i)
        {
            //INFO// cout << "Cami cap a omega: " << from.layer+1 << ", " << from.numO << ", " << i+1 << endl;
            sum_derivate += derivate_omega(nexusWeb, wireWeb, to, wireWeb[from.layer][from.numO-1][i], pas+1);
        }

        return (init * sum_derivate);
    }
}

//-------------------------------------------------------

double neuron_derivate(const Nexus& neuron)
{
    return (neuron.alpha * (1 - neuron.alpha));
}

//-------------------------------------------------------

void print_outputs(const NexusWeb& nexusWeb, const DataVector& expectedOutputs)
{
    for(unsigned int i = 0; i < nexusWeb[nexusWeb.size()-1].size(); ++i)
    {
        cout << "OUTPUT <" << i+1 << ">: " << nexusWeb[nexusWeb.size()-1][i].alpha << ", (EXPECTED: " << expectedOutputs[i] << ")" << endl;
    }
}

//-------------------------------------------------------

void print_info(const NexusWeb& nexusWeb, const WireWeb& wireWeb)
{
    cout << "----------------------------------------------------------------------------" << endl;
    
    cout << " ## INFO ## " << endl;
    cout << "------------" << endl;

    cout << "Num. Nexus layers: " << nexusWeb.size() << endl;
    cout << "Num. Wire layers: " << wireWeb.size() << endl << endl;
    
    cout << "----------------------------------------" << endl;
    
    for(unsigned int i = 0; i < nexusWeb.size(); ++i) 
    {
        cout << "|Nexus layer [" << i+1 << "]:" << endl;
        
        for(unsigned int j = 0; j < nexusWeb[i].size(); ++j)
        {
            cout << "|---|> Nexus [" << j+1 << "]:";
            
            if(nexusWeb[i][j].p) cout << " *INPUT*" << endl;
            else cout << endl;

            cout << "|   |# alpha = " << nexusWeb[i][j].alpha << endl;
            if(not nexusWeb[i][j].p)cout << "|   |# u = " << nexusWeb[i][j].u << endl;
            
            if(not nexusWeb[i][j].p)
            {
                for(unsigned int k = 0; k < nexusWeb[i-1].size(); ++k)
                {
                    cout << "|   |---|-# Wire from [" << k+1 << " (layer " << i << ")]:" << endl;
                    cout << "|   |   |# omega = " << wireWeb[i-1][k][j].omega << endl;
                    cout << "|   |   |# *alpha = " << nexusWeb[i-1][k].alpha << endl;
                }
            }
            else
            {
                cout << "|   |---|# Input <" << nexusWeb[i][j].num << "> = " << nexusWeb[i][j].alpha << endl;
            }

            cout << "|" << endl;
        }
        cout << "----------------------------------------" << endl;
    }
    
    cout << "----------------------------------------------------------------------------" << endl;
}

//-------------------------------------------------------

void test_inputs(NexusWeb& nexusWeb, const WireWeb& wireWeb)
{
    DataVector input(nexusWeb[0].size());
    DataVector output(nexusWeb[nexusWeb.size()-1].size());

    cout << "----------------------------------------" << endl;
    cout << "TEST PHASE:" << endl << endl;
    cout << "Enter custom inputs: " << endl;
    for(unsigned int i = 0; i < nexusWeb[0].size(); ++i)
    {
        cout << "X" << i+1 << ": ";
        cin >> input[i];
    } 

    cout << "Enter custom outputs: " << endl;
    for(unsigned int i = 0; i < nexusWeb[nexusWeb.size()-1].size(); ++i)
    {
        cout << "Y" << i+1 << ": ";
        cin >> output[i];
    }

    activate(nexusWeb, wireWeb, input);

    print_outputs(nexusWeb, output);

    cout << "-------------------" << endl;

}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------//