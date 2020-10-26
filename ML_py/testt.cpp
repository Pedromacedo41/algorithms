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

vi reponses;


int main(){
    int testcases;
    cin >> testcases;
    f(i,testcases){
        int n,x,a,b;
        int res=0;
        cin >> n >> x >> a>> b;
        int c,d;
        c = min(a,b);
        d= (c==a)? b:a;
        //cout << "(" << c << "," << d << ")" << endl;
        int mar= c-1+ (n-d);
        // cout << mar << endl;
        int dis = d-c;
        // cout << dis << endl;
        int opt= min(x,mar);
        //cout << opt << endl;
        reponses.pb(dis+opt);
    }
    f(i,testcases){
        cout << reponses[i] << endl;
    }
    return 0;
}
