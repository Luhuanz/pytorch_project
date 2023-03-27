// studying 2023-3-27
//Created by luke
// 变量基本类型


#include <iostream>
#include "Sales_item.h"
#include<string>
void basic_class(){
    Sales_item book;
    std::cin >> book;
    std::cout << book<<std::endl;
}
void q_2_3()
{
    unsigned u1 = 10, u2 = 42;
    std::cout << u2 - u1 << std::endl;
    std::cout << u1 - u2 << std::endl;
    int i = 10, i2 = 42;
    std::cout << i2 - i << std::endl;
    std::cout << i2 - u1 << std::endl;
    std::cout << i - u2 << std::endl;
}

struct Sale_data
{
    std::string bookNo;
    unsigned units_sold=0;
    double revenue=0.0;
};
int q1(){
    Sale_data book;
    double price;
    std::cin>>book.bookNo>>book.units_sold>>price;
    book.revenue=book.units_sold*price;
    std::cout<<book.bookNo<<" 价格"<<book.revenue<<std::endl;



    return 0;
}

int q2(){
Sale_data book1,book2;
double price1,price2;
std::cout<<"请输入书本1:"<<std::endl;
std::cin>>book1.bookNo>>book1.units_sold>>price1;
std::cout << "请输入书本2:" << std::endl;
std::cin >> book2.bookNo >> book2.units_sold >> price2;
book1.revenue = book1.units_sold * price1;
book2.revenue = book2.units_sold * price2;
if(book1.bookNo==book2.bookNo){
    unsigned tC=book1.units_sold+book2.units_sold;
    double toR=book1.revenue+book2.revenue;
    std::cout<<book1.bookNo<< " "<<tC<<" "<<toR<<std::endl;
    



}

}


int main(){

// basic_class();
// q_2_3();
// q1();

    return 0;
}