#include <bits/stdc++.h>

using namespace std;

vector<string> split_string(string);

// Complete the sockMerchant function below.
int sockMerchant(int n, vector<int> ar) {
    unordered_map<int, int> pairs;
    int d=0;
    for(auto e : ar){
        if(pairs.find(e)== pairs.end()){
            pairs.insert(make_pair(0,0));
        }
        else{
            pairs[e]+=1;
        }
    }
    
    //for(auto e : pairs) d+= floor(e.second/2);
    for(auto e : pairs) cout << e.first << " " <<  e.second << endl;
    return d;
}

int main(){
  
    

}