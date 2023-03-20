#include <ostream>
#include "Sales_item.h"

void basic_io()
{
    std::cout << "Enter two numbers1:" << std::endl;
    int v1, v2;
    std::cin >> v1 >> v2;
    std::cout << "The sum of " << v1 << " and " << v2
              << " is " << v1 + v2 << std::endl;
}

void basic_while()
{
    int sum = 0;
    int val = 1;

    while (val <= 10)
    {
        sum += val;
        ++val;
    }
    std::cout << "The sum of " << sum << std::endl;
}

void basic_for()
{
    int sum = 0;
    for (int i = 0; i < 10; i++)
    {
        sum += i;
    }
    std::cout << "The sum of " << sum << std::endl;
}

void basic_if()
{
    std::cout << "Enter the number of : " << std::endl;
    int v1, v2;
    std::cin >> v1 >> v2;
    int lower, upper;
    if (v1 <= v2)
    {
        lower = v1;
        upper = v2;
    }
    else
    {
        lower = v2;
        upper = v1;
    }
    int sum = 0;
    for (int val = lower; val <= upper; ++val)
        sum += val;
    std::cout << "Sum of " << lower
              << " to " << upper
              << " inclusive is "
              << sum << std::endl;
}

void basic_cin()
{
    int sum = 0, value;
    std::cout << "Enter the number :" << std::endl;
    while (std::cin >> value)
    {
        sum += value;
    }
    std::cout << "Sum of " << sum << std::endl;
}

void basic_cout()
{
    int sum = 0;
    for (int i = 0; std::cin >> i;)
    {
        sum += i;
    }
    std ::cout << "Enter the sum " << sum << std::endl;
}

int main()
{
    // basic_io();
    //  basic_while();
    //  basic_for();
    //  basic_if();
    // basic_cin();
    basic_cout();
    return 0;
}