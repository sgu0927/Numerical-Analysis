#include <iostream>
#include <stdio.h>
#include <cmath>
using namespace std;

double coefficient[5] = { -23.4824832, 24.161472, 15.85272, -22.4, 5.0 };
double differential[4] = { 24.161472, 31.70544, -67.2, 20.0 };
//polynomial equation
double func(double x) {
	double res = 0;
	for (int i = 0; i < 5; i++) {
		res += coefficient[i] * (pow(x, i));
	}
	return res;
}
//derivative
double derivative(double x) {
	double res = 0;
	for (int i = 0; i < 4; i++) {
		res += differential[i] * (pow(x, i));
	}
	return res;
}

int main(void)
{
	// Bisection Algorithm
	double left = 1.0, right = 1.3, mid;
	double diff = 0.00001, ans = 987654321;
	scanf_s("%lf %lf", &left, &right);
	while (true) {
		mid = (left + right) / 2;
		if (abs(right-mid) < diff) {
			ans = mid;
			break;
		}
		if (func(left) * func(mid) < 0) {
			right = mid;
		}
		else {
			left = mid;
		}
	}
	printf("%lf", ans);

	//Newton Raphson Method
	/*double initial_guess = 1.0, diff = 0.0001, ans = 987654321;
	scanf_s("%lf", &initial_guess);

	while (true) {
		if (abs(func(initial_guess) / derivative(initial_guess)) < diff) {
			ans = initial_guess;
			break;
		}
		else {
			initial_guess -= func(initial_guess) / derivative(initial_guess);
		}
	}
	printf("%lf", ans);*/

	return 0;
}