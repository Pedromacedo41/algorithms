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
    
    prim_map[9]++;
    cout <<prim_map[9] << endl;
    return 0;
}