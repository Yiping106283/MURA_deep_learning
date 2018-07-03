#include<iostream>
#include<string.h>
#include<stdlib.h>
#include<fstream>
#include<stdio.h>
#include<string>
using namespace std;
#define len 1920 
int main(){
    FILE* fp;
    FILE* fquery;
    char query_dir[20];
    fp = fopen("out.txt","r");
    fquery = fopen("query_weight.txt", "r");
    char str[150];
    char minpath[150], path[150];
    float mindis, dis;
    mindis = 1e10;
    float weight[2000]={};
    for(int i=0;i<len;++i){
        fscanf(fquery, "%s", str);
        weight[i]=atof(str);
    }
    int cnt=0;
    fclose(fquery);
    float tmp = 0.0;
    while(fscanf(fp, "%s", str)!=EOF){
//        cnt++;cout<<cnt<<endl;
        memcpy(path, str, sizeof(str));
        dis = 0;
        for(int i=0;i<len;++i){
            fscanf(fp, "%s", str);
            tmp = atof(str)-weight[i];
            dis += (tmp*tmp);
        }
        if(dis < mindis){
            memcpy(minpath, path, sizeof(path));
            mindis = dis;
        }
  //      cout<<dis<<endl;
    }
    fclose(fp);
    cout<<minpath<<endl;
//    cout<<mindis<<endl;
    return 0;
}
