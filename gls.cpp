
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "math.h"
#include "string"

struct city
{
    double x;
    double y;
};

std::string getCmdOption(int argc, char* argv[], const std::string& option) {

    std::string cmd;

    for (int i = 0; i < argc; i++) {

        std::string arg = argv[i];
        if (arg.find(option) == 0) {

            std::size_t found = arg.find_last_of(option);
            cmd = arg.substr(found + 1);
            return cmd;
        }
    }

    return cmd;
}

double euc2d(const city& city1, const city& city2) {

    return sqrt(pow(city1.x - city2.x, 2) + pow(city1.y - city2.y, 2));
}

auto generateEmptyPenaltyMatrix(int size) {

    std::vector<std::vector<double>> penaltyMatrix;
    for (int i = 0; i < size; i++) {
        std::vector<double> row;
        for (int j = 0; j < size; j++) {
            row.push_back(0.0);
        }
        penaltyMatrix.push_back(row);
    }

    return penaltyMatrix;
}

auto getDistanceMatrix(const std::vector<city> cities) {

    std::vector<std::vector<double>> distanceMatrix;

    for (const auto myCity : cities) {

        std::vector<double> row;
        for (const auto myCity2 : cities) {

            double distance = euc2d(myCity, myCity2);
            row.push_back(distance);
        }
        distanceMatrix.push_back(row);
    }

    return distanceMatrix;
}

std::vector<city> readProblem(const std::string& filename, int& nCities) {

    std::string line;
    std::ifstream myFile;
    myFile.open(filename);

    if (!myFile.is_open()) {
        perror("Error open");
        exit(EXIT_FAILURE);
    }

    std::vector<city> myCities;
    bool firstLine = true;

    while (std::getline(myFile, line)) {

        std::stringstream ss(line);

        if (!firstLine) {

            city myCity;
            ss >> myCity.x >> myCity.y;
            myCities.push_back(myCity);
        }

        else {

            ss >> nCities;
            firstLine = false;
        }
    }

    return myCities;
}

void printSolution(const std::vector<int>& solution) {

    for (int i = 0; i < solution.size(); i++)
        std::cout << solution[i] << " ";
    std::cout << std::endl;
}

void saveSolution(const std::vector<int>& solution, const double& solutionCost, const std::string& filename) {

    std::ofstream out(filename);
    out << solutionCost << std::endl;
    for (int i = 0; i < solution.size(); i++)
        out << solution[i] << " ";
    out.close();
}

void saveCostEvolution(const std::vector<double>& costEvolution, const std::string& filename) {

    std::ofstream out(filename);
    out << "it,cost" << std::endl;
    for (int i = 0; i < costEvolution.size(); i++)
        out << i << "," << costEvolution[i] << std::endl;
    out.close();
}

double cost(const std::vector<int>& solution, const std::vector<std::vector<double>>& distanceMatrix) {

    double c = 0;
    for (unsigned int i = 0; i < solution.size(); i++) {

        int j = (i + 1) % solution.size();
        int u = solution[i];
        int v = solution[j];

        c += distanceMatrix[u][v];
    }

    return c;
}

double augmentedCost(const std::vector<int>& solution, const std::vector<std::vector<double>>& distanceMatrix,
    const std::vector<std::vector<double>>& penalties, double lambda) {

    double c = 0;
    for (unsigned int i = 0; i < solution.size(); i++) {

        int j = (i + 1) % solution.size();
        int u = solution[i];
        int v = solution[j];

        c += distanceMatrix[u][v];
        c += lambda * penalties[u][v];
    }

    return c;
}

auto _2opt(const std::vector<int>& solution, std::vector<int>& newSolution, int i, int j) {

    newSolution.clear();

    for (int t = 0; t < i; t++)
        newSolution.push_back(solution[t]);

    for (int t = j; t >= i; t--)
        newSolution.push_back(solution[t]);

    for (unsigned int t = j + 1; t < solution.size(); t++)
        newSolution.push_back(solution[t]);
}

auto generateCandidate(const std::vector<int>& solution, std::vector<int>& newSolution) {

    int i = rand() % solution.size();
    int j = rand() % solution.size();

    j = (i == j) ? (j + 1) % solution.size() : j;

    if (i > j) {

        int temp = i;
        i = j;
        j = temp;
    }

    _2opt(solution, newSolution, i, j);
}

void updatePenalties(const std::vector<int>& solution, const std::vector<std::vector<double>>& distanceMatrix,
    std::vector<std::vector<double>>& penalties) {

    int maxU = solution[0];
    int maxV = solution[1];
    double maxUtility = distanceMatrix[maxU][maxV] / (1 + penalties[maxU][maxV]);

    for (unsigned int i = 1; i < solution.size(); i++) {

        int j = (i + 1) % solution.size();
        int u = solution[i];
        int v = solution[j];

        double utility = distanceMatrix[u][v] / (1 + penalties[u][v]);

        if (utility > maxUtility) {
            
            maxUtility = utility;
            maxU = u;
            maxV = u;
        }
    }

    penalties[maxU][maxV]++;
}

void localSearch(std::vector<int>& solution, const std::vector<std::vector<double>>& distanceMatrix,
    const std::vector<std::vector<double>>& penalties, double lambda, int maxAttemps) {

    int attemps = 0;
    double bestCost = augmentedCost(solution, distanceMatrix, penalties, lambda);
    std::vector<int> candidate;

    while (attemps < maxAttemps) {

        generateCandidate(solution, candidate);

        double costCandidate = augmentedCost(candidate, distanceMatrix, penalties, lambda);

        if (costCandidate < bestCost) {

            solution = candidate;
            bestCost = costCandidate;
        }

        else {

            attemps++;
        }
    }
}

std::vector<int> gls(const std::vector<int>& solution, const std::vector<std::vector<double>>& distanceMatrix,
    std::vector<std::vector<double>>& penalties, double alpha, int maxAttemps, int maxIterations, std::vector<double>& costEvolution) {

    int count = 0;

    std::vector<int> bestSolution = solution;
    double bestCost = cost(solution, distanceMatrix);
    double lambda = alpha * bestCost / solution.size();

    std::vector<int> current = solution;
    costEvolution.push_back(bestCost);

    while (count < maxIterations) {

        localSearch(current, distanceMatrix, penalties, lambda, maxAttemps);
        double currentCost = cost(current, distanceMatrix);

        if (currentCost < bestCost) {

            bestCost = currentCost;
            bestSolution = current;
            std::cout << "Best solution found with cost " << bestCost << std::endl;
            printSolution(bestSolution);
        }

        updatePenalties(current, distanceMatrix, penalties);
        count++;

        costEvolution.push_back(bestCost);
    }

    return bestSolution;
}

std::vector<int> getInitialSolution(const int& nCities) {

    std::vector<int> initialSolution;

    for (int i = 0; i < nCities; i++)
        initialSolution.push_back(i);
    return initialSolution;
}


int main(int argc, char* argv[]) {

    // Parameteres
    int maxAttemps = std::atoi(getCmdOption(argc, argv, "-maxAttemps=").c_str());
    int maxIterations = std::atoi(getCmdOption(argc, argv, "-maxIterations=").c_str());
    double alpha = std::atof(getCmdOption(argc, argv, "-alpha=").c_str());

    // Read the problem
    int nCities;
    std::string problem = getCmdOption(argc, argv, "-f=");
    std::cout << "Solving problem " << problem << std::endl;
    std::vector<city> myCities = readProblem(problem, nCities);

    // Define a distance matrix
    std::vector<std::vector<double>> distanceMatrix = getDistanceMatrix(myCities);

    // Define a penalty matrix
    std::vector<std::vector<double>> penaltyMatrix = generateEmptyPenaltyMatrix(nCities);

    // Get a initial solution
    std::vector<int> initialSolution = getInitialSolution(nCities);
    std::vector<double> costEvolution;

    std::cout << "Cost of initial solution is " << cost(initialSolution, distanceMatrix) << std::endl;

    std::vector<int> bestSolution = gls(initialSolution, distanceMatrix, penaltyMatrix, alpha, maxAttemps, maxIterations, costEvolution);
    double costSolution = cost(bestSolution, distanceMatrix);
    std::cout << "Final solution cost is " << costSolution << std::endl;
    printSolution(bestSolution);
    saveSolution(bestSolution, costSolution, "solution.txt");
    saveCostEvolution(costEvolution, "cost_evolution.txt");

    return 0;
}