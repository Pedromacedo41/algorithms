#include <bits/stdc++.h>
using namespace std;

#define st first
#define nd second
#define mp make_pair
#define pb push_back
#define cl(x, v) memset((x), (v), sizeof(x))
#define f(i,n) for(int i=0; i < n; i++)

typedef long long ll;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<int, pii> piii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;

vi primes;
unordered_map<int,int> prim_map;
vector<pii> prim;

int main(){
    int testcases;
    cin >> testcases;
    f(i,testcases){
        int x;
        cin >> x;
        primes.pb(x);
    }
    for(auto e: primes){
        prim_map[e]++;
    }
    for(auto e: prim_map){
        prim.pb(mp(e.nd,e.st));
    }
    sort(prim.begin(), prim.end());
    /*
    for(auto e: prim){
        cout << e.st << " " << e.nd << endl;
    }*/
    int tot= prim.size();
    f(i,prim.size()){
        tot+=(prim[i].st-1)*(prim.size()-1-i);
    }
    cout << tot << endl;
    return 0;
}