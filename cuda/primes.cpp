#include "primes.h"

//returns vector with pairs of <primefactor, multiplicity of that factor> for a given number
std::vector<std::pair<unsigned int, unsigned int>> primeFactorization(unsigned int number){

	std::vector<std::pair<unsigned int, unsigned int>> primes; //prime number and multiplicity
	primes.push_back(std::make_pair(1, 1)); //highly disputable, but is needed for the following primeCombination-funtions

	for(unsigned int i = 2; i*i <= number; i++){ // brute force, but only necessary up to at most sqrt(number)
		unsigned int count = 0;
		while(number%i == 0){
			number /= i;
			count++;
		}
		if(count != 0){
			primes.push_back(std::make_pair(i, count));
		}
	}
	if(number != 1){ //if number != 1 then itself is a prime number
		primes.push_back(std::make_pair(number, 1));
	}
	return primes;
}

//returns all possible combinations of the primefactors of a given number
std::vector<unsigned int> primeCombinations(unsigned int number){

	std::vector<std::pair<unsigned int, unsigned int>> primes = primeFactorization(number);

	std::vector<std::vector<unsigned int>> singlePrimeCombinations;
	for(std::pair<unsigned int, unsigned int> primePair : primes){
		unsigned int prime = primePair.first;
		unsigned int count = primePair.second;
		std::vector<unsigned int> singlePrime;
		unsigned factor = 1;
		for(unsigned int c = 0; c < count; c++){
			factor *= prime;
			singlePrime.push_back(factor);
		}
		if(prime != 1){
			singlePrime.insert(singlePrime.begin(), 1);
		}
		singlePrimeCombinations.push_back(singlePrime);
	}

	std::vector<unsigned int> factor1Vec = singlePrimeCombinations[0]; //primeFactorization returns at least (1, 1) in the vector, so the upper loop is traversed at least once
	for(unsigned int otherPrimes = 1; otherPrimes < singlePrimeCombinations.size(); otherPrimes++){
		std::vector<unsigned int> tmpfactor1Vec;
		for(unsigned int c = 0; c < singlePrimeCombinations[otherPrimes].size(); c++){
			for(unsigned int i = 0; i < factor1Vec.size(); i++){
				tmpfactor1Vec.push_back(factor1Vec[i]*singlePrimeCombinations[otherPrimes][c]);
			}
		}
		factor1Vec = tmpfactor1Vec;
	}

	return factor1Vec;
}

//returns all possible 2D-combinations of the primefactors of the number, i.e. all pairs of (x, y) s.t. x*y=number
std::vector<std::pair<unsigned int, unsigned int>> primeCombinations2D(unsigned int number){

	std::vector<unsigned int> factor1Vec = primeCombinations(number);

	std::vector<std::pair<unsigned int, unsigned int>> combinations;
	for(unsigned int factor1 : factor1Vec){
		//combinations.push_back(std::make_pair(factor1, total/factor1));
		combinations.push_back(std::make_pair(factor1, number/factor1));
	}
	return combinations;
}

//same but in 3D, tuples of (x, y, z) s.t. x*y*z=number
std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> primeCombinations3D(unsigned int number){

	std::vector<unsigned int> factor1Vec = primeCombinations(number);

	std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> combinations;
	for(unsigned int factor1 : factor1Vec){

		std::vector<std::pair<unsigned int, unsigned int>> combinations2D = primeCombinations2D(number/factor1);

		for(std::pair<unsigned int, unsigned int> pair2D : combinations2D){
			combinations.push_back(std::make_tuple(factor1, pair2D.first, pair2D.second));
		}
	}
	return combinations;
}
