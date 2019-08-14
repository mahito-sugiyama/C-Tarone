#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <functional>
#include <utility>
#include <iomanip>
#include <set>
#include <cstdint>
#include <unistd.h>
#include <boost/math/distributions/chi_squared.hpp>

#define Int int32_t
Int IDX_FEATURE;
double NUM_PATTERN;
bool VERBOSE = false;
double SIGMA = 0.2;
double ADMISSIBLE;
double FWER_ESTIMATE;
double ZERO = 1e-200;

using namespace std;

struct cmp {
  bool operator ()(const pair<double, double> &a, const pair<double, double> &b) {
    // return a.second < b.second;
    return a.first > b.first;
  }
};

// output a 2D vector
template <typename T>
ostream &operator<<(ostream& out, const vector<vector<T>>& mat) {
  for (Int i = 0; i < mat.size() - 1; i++) {
    for (auto&& x : mat[i]) {
      out << x << " ";
    }
    out << endl;
  }
  for (auto&& x : mat.back()) {
    out << x << " ";
  }
  return out;
}
// output a vector
template <typename T>
ostream &operator<<(ostream& out, const vector<T>& vec) {
  if (vec.size() == 0) return out;
  for (Int i = 0; i < vec.size() - 1; ++i) {
    out << vec[i] << " ";
  }
  out << vec.back();
  return out;
}
// output a set
template <typename T>
ostream &operator<<(ostream& out, const set<T>& vec) {
  auto first = vec.begin();
  auto last = vec.empty() ? vec.end() : prev(vec.end()); // in case vec is empty
  for (auto it = first; it != last; ++it) {
    out << *it << ", ";
  }
  out << *(prev(vec.end()));
  return out;
}

// read a database file
void readFromCSV(ifstream& ifs, vector<vector<double>>& data) {
  string line;
  while (getline(ifs, line)) {
    stringstream lineStream(line);
    string cell;
    vector<double> tmp;
    while (getline(lineStream, cell, ',')) {
      tmp.push_back(stod(cell));
    }
    data.push_back(tmp);
  }
}
void readClassFromCSV(ifstream& ifs, vector<Int>& cl) {
  string line;
  while (getline(ifs, line)) {
    cl.push_back(stoi(line));
  }
}

// compute frequency
double computeFreq(vector<vector<double>>& rankn, vector<Int>& fset) {
  Int N = rankn.size();
  double freq = 0.0;
  if (fset.size() == 0) return 1.0;

  for (Int i = 0; i < N; ++i) {
    double freq_each = 1.0;
    for (auto&& j : fset) {
      freq_each *= rankn[i][j];
    }
    freq += freq_each;
  }
  freq /= (double)N;

  return freq;
}
// compute frequency using the previous result
double computeFreqUpdate(vector<vector<double>>& rankn, Int fnew, vector<double>& freq_current) {
  Int N = rankn.size();
  double freq = 0.0;
  // if (fset.size() == 0) return 1.0;
  for (Int i = 0; i < N; ++i) {
    freq_current[i] *= (double)rankn[i][fnew];
    freq += freq_current[i];
  }
  freq /= (double)N;
  return freq;
}

// compute the maximum frequency (eta)
double eta_max(Int k, Int N, Int N_class) {
  double sum = 0.0;
  for (Int i = 1; i <= N_class; ++i) {
    sum += pow((double)(N - i + 1) / (double)N, (double)k);
  }
  sum /= (double)N;
  return sum;
}
// compute the mininum frequency (eta)
double eta_min(Int k, Int N, Int N_class) {
  vector<Int> idx(N_class);
  iota(idx.begin(), idx.end(), 1);
  // ascending order vector
  vector<double> ascending_order(N_class);
  for (Int i = 0; i < N_class; ++i) {
    ascending_order[i] = (double)idx[i] / (double)N;
  }
  // current value
  vector<double> v;
  copy(ascending_order.begin(), ascending_order.end(), back_inserter(v));

  for (Int j = 0; j < k - 1; ++j) {
    sort(v.begin(), v.end(), greater<double>());
    for (Int i = 0; i < N_class; ++i) {
      v[i] *= ascending_order[i];
    }
  }
  double sum = accumulate(v.begin(), v.end(), 0.0);
  sum /= (double)N;
  return sum;
}
// compute the maximum achievable KL divergence
double kl_max(double freq, Int N0, Int N) {
  double r0 = (double)N0 / (double)N;
  if (freq < r0) {
    if (freq < ZERO) {
      return (r0 - freq) * log((r0 - freq) / (r0 - r0*freq)) + (1 - r0) * log(1 / (1 - freq));
    } else if (fabs(r0 - freq) < ZERO) {
      return freq * log(1 / r0) + (1 - r0) * log(1 / (1 - freq));
    } else {
      return freq * log(1 / r0) + (r0 - freq) * log((r0 - freq) / (r0 - r0*freq)) + (1 - r0) * log(1 / (1 - freq));
    }
  } else {
    if (fabs(freq - r0) < ZERO) {
      return r0 * log(1 / freq) + (1 - freq) * log(1 / (1 - r0));
    } else if (fabs(1 - freq) < ZERO) {
      return r0 * log(1 / freq) + (freq - r0) * log((freq - r0) / (freq - freq*r0));
    } else {
      return r0 * log(1 / freq) + (freq - r0) * log((freq - r0) / (freq - freq*r0)) + (1 - freq) * log(1 / (1 - r0));
    }
  }
}
double kl(vector<vector<double>>& rankn, vector<Int>& fset, vector<Int>& cl, double freq, Int N0, Int N) {
  double r0 = (double)N0 / (double)N;
  double r1 = (double)(N - N0) / (double)N;
  double freq0 = 0.0;

  for (Int i = 0; i < N; ++i) {
    if (cl[i] == 0) {
      double freq_each = 1.0;
      for (auto&& j : fset) {
	freq_each *= rankn[i][j];
      }
      freq0 += freq_each;
    }    
  }
  freq0 /= (double)N;
  double freq1 = freq - freq0;

  vector<double> po;
  po.push_back(freq0);
  po.push_back(freq1);
  po.push_back(r0 - freq0);
  po.push_back(r1 - freq1);

  vector<double> pe;
  pe.push_back(r0 * freq);
  pe.push_back(r1 * freq);
  pe.push_back(r0 - r0 * freq);
  pe.push_back(r1 - r1 * freq);
 
  double kl = 0.0;
  for (Int i = 0; i < po.size(); ++i) {
    kl += po[i] * log(po[i] / pe[i]);
  }
  return kl;
}
// compute p-value
double computePvalue(double kl, Int N) {
  boost::math::chi_squared chisq_dist(1);
  // else pval = 1 - boost::math::cdf(chisq_dist, 2 * (double)N * kl);
  // if (pval > 1) pval = 1.0;
  // if (VERBOSE) cout << "kl: " << kl << endl;
  double pval = 0.0;
  if (kl <= pow(10, -8)) pval = 1.0;
  else pval = 1 - boost::math::cdf(chisq_dist, 2 * (double)N * kl);
  return pval;
}
// compute a list of thresholds
void computeThresholds(vector<double>& freq_thrs, Int n, Int N) {
  freq_thrs.push_back(eta_max(1, N, N));
  for (Int k = n; k >= 2; --k) {
    freq_thrs.push_back(eta_max(k, N, N));
    freq_thrs.push_back(eta_min(k, N, N));
  }
  sort(freq_thrs.begin(), freq_thrs.end(), greater<double>());
}

// Frequent Pattern Mining
void runFPM(vector<vector<double>>& rankn, vector<Int>& fset, vector<double>& freq_current, Int i_prev, Int n, Int size_limit, ofstream& ofs) {
  Int N = rankn.size();
  for (Int i = i_prev + 1; i < n; i++) {
    fset.push_back(i);
    double freq = computeFreq(rankn, fset);
    if (freq > SIGMA && fset.size() <= size_limit) {
      if (VERBOSE) cout << "  Copula support of {" << fset << "} = " << freq << endl;
      // ofs << fset << " (" << freq << ")" << endl;
      NUM_PATTERN += 1.0;
      runFPM(rankn, fset, freq_current, i, n, size_limit, ofs);
    }
    // extract the feature fset.back();
    /*
    for (Int i = 0; i < N; ++i) {
      freq_current[i] /= (double)rankn[i][fset.back()];
    }
    */
    fset.pop_back();
  }
}
// Significant Pattern Mining
void runSPM(vector<vector<double>>& rankn, vector<Int>& fset, vector<double>& freq_current, Int i_prev, Int n, Int size_limit, ofstream& ofs, Int N0, set<pair<double, double>, cmp>& freq_pval_list, double alpha) {
  Int N = rankn.size();
  for (Int i = i_prev + 1; i < n; i++) {
    fset.push_back(i);
    double freq = computeFreq(rankn, fset);
    if (freq > SIGMA && fset.size() <= size_limit) {
      NUM_PATTERN += 1.0;
      if (VERBOSE) cout << "  Copula support of {" << fset << "} = " << freq << endl;
      // if (freq < (double)N0 / (double)N) {
      double pval = computePvalue(kl_max(freq, N0, N), N);
      freq_pval_list.insert(make_pair(freq, pval));
      ADMISSIBLE = alpha / (*prev(freq_pval_list.end())).second;
      while (ADMISSIBLE < NUM_PATTERN) {
	// SIGMA = (*prev(freq_pval_list.end())).first;
	SIGMA = min((double)N0 / (double)N, (*prev(freq_pval_list.end())).first);
	// cout << "update sigma:" << SIGMA << endl;
	freq_pval_list.erase(prev(freq_pval_list.end()));
	NUM_PATTERN -= 1.0;
	ADMISSIBLE = freq_pval_list.empty() ? 1e20 : alpha / (*prev(freq_pval_list.end())).second;
      }
      // }
      // ofs << fset << " (" << freq << ")" << endl;
      if (VERBOSE) cout << "  Current copula support threshold =    " << SIGMA << endl;
      if (VERBOSE) cout << "  Current admissible number = " << ADMISSIBLE << endl;
      runSPM(rankn, fset, freq_current, i, n, size_limit, ofs, N0, freq_pval_list, alpha);
    }
    // extract the feature fset.back();
    /*
    for (Int i = 0; i < N; ++i) {
      freq_current[i] /= (double)rankn[i][fset.back()];
    }
    */
    fset.pop_back();
  }
}
// Frequent Pattern Mining with finding singificant patterns
void runSPM_sig(vector<vector<double>>& rankn, vector<Int>& fset, vector<double>& freq_current, Int i_prev, Int n, Int size_limit, Int N0, vector<Int>& cl, double alpha_corrected, ofstream& ofs) {
  Int N = rankn.size();
  for (Int i = i_prev + 1; i < n; i++) {
    fset.push_back(i);
    double freq = computeFreq(rankn, fset);
    if (freq > SIGMA && fset.size() <= size_limit) {
      double pval = computePvalue(kl(rankn, fset, cl, freq, N0, N), N);
      // if (VERBOSE) cout << "  corrected p-value of {" << fset << "} = " << pval * (double)num_testable << endl;
      if (pval < alpha_corrected) {
	NUM_PATTERN += 1.0;
	// ofs << fset << " (" << pval * (double)num_testable << ")" << endl;
	ofs << fset << " (" << pval << ")" << endl;
      }
      runSPM_sig(rankn, fset, freq_current, i, n, size_limit, N0, cl, alpha_corrected, ofs);
    }
    // extract the feature fset.back();
    /*
    for (Int i = 0; i < N; ++i) {
      freq_current[i] /= (double)rankn[i][fset.back()];
    }
    */
    fset.pop_back();
  }
}

int main(int argc, char *argv[]) {
  bool flag_in = false;
  bool flag_class_in = false;
  bool flag_out = false;
  bool flag_stat = false;
  bool fpm = false;
  bool bonferroni = false;
  char *input_file = NULL;
  char *input_class_file = NULL;
  char *output_file = NULL;
  char *stat_file = NULL;
  Int size_limit = INT32_MAX;
  double alpha = 0.05;
  clock_t ts, te;

  // get arguments
  char opt;
  while ((opt = getopt(argc, argv, "i:c:o:t:s:a:k:fvwp:b")) != -1) {
    switch (opt) {
    case 'i': input_file = optarg; flag_in = true; break;
    case 'c': input_class_file = optarg; flag_class_in = true; break;
    case 'o': output_file = optarg; flag_out = true; break;
    case 't': stat_file = optarg; flag_stat = true; break;
    case 's': SIGMA = atof(optarg); break;
    case 'a': alpha = atof(optarg); break;
    case 'k': size_limit = atoi(optarg); break;
    case 'f': fpm = true; break;
    case 'v': VERBOSE = true; break;
    case 'b': bonferroni = true; break;
    }
  }

  if (!flag_in) {
    cerr << "> ERROR: Input file (-i [input_file]) is missing!" << endl;
    exit(1);
  }
  if (!flag_out) {
    output_file = (char *)"output";
  }
  if (!flag_stat) {
    stat_file = (char *)"stat";
  }
  ofstream sfst(stat_file);

  // --------------------------------- //
  // ---------- Read a file ---------- //
  // --------------------------------- //
  cout << "> Reading a database file     \"" << input_file << "\" ... " << flush;
  sfst << "> Reading a database file     \"" << input_file << "\" ... ";
  ifstream ifs(input_file);
  vector<vector<double>> data;
  vector<Int> cl;
  readFromCSV(ifs, data);
  cout << "end" << endl << flush;
  sfst << "end" << endl;
  Int N = data.size();
  Int n = data[0].size();
  Int N0 = 0;
  if (!fpm) {
    cout << "> Reading a class file        \"" << input_class_file << "\" ... " << flush;
    sfst << "> Reading a class file        \"" << input_class_file << "\" ... ";
    ifstream cfs(input_class_file);
    readClassFromCSV(cfs, cl);
    cout << "end" << endl << flush;
    sfst << "end" << endl;
    for (auto&& x : cl) if (x == 0) N0++;
    if (N0 > N / 2) {
      cerr << "> ERROR: Class 0 must be a minor class!" << endl;
      exit(1);
    }
  }
  cout << "  Sample size in total:       " << N << endl << flush;
  sfst << "  Sample size in total:       " << N << endl;
  if (!fpm) {
    cout << "  Sample size in class 0:     " << N0 << endl << flush;
    sfst << "  Sample size in class 0:     " << N0 << endl;
  }
  cout << "  # features:                 " << n << endl << flush;
  sfst << "  # features:                 " << n << endl;
  if (fpm) {
    cout << "  copula support threshold:     " << SIGMA << endl << flush;
    sfst << "  copula support threshold:     " << SIGMA << endl;
  }

  // ----------------------------------- //
  // ---------- Compute ranks ---------- //
  // ----------------------------------- //
  vector<vector<Int>> rank;
  rank = vector<vector<Int>>(N, vector<Int>(n, 0));
  vector<vector<double>> rankn;
  rankn = vector<vector<double>>(N, vector<double>(n, 0.0));
  vector<size_t> idx(N);
  for (Int j = 0; j < n; ++j) {
    // initialize index vector
    iota(idx.begin(), idx.end(), 0);
    IDX_FEATURE = j;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
	 [&data](Int i1, Int i2) {return data[i1][IDX_FEATURE] < data[i2][IDX_FEATURE];});
    for (Int i = 0; i < N; i++) {
      rank[idx[i]][j] = i;
      rankn[idx[i]][j] = (double)rank[idx[i]][j] / ((double)N - 1.0);
    }
  }



  // --------------------------------- //
  // ---------- Enumeration ---------- //
  // --------------------------------- //
  if (VERBOSE) {
    cout << "> Start enumeration of testable combinations:" << endl << flush;
  } else {
    cout << "> Start enumeration of testable combinations ... " << flush;
    sfst << "> Start enumeration of testable combinations ... ";
  }
  vector<double> freq_current(N, 1.0); // track the current rank product for each sample
  vector<Int> fset; // pattern (a set of features)
  ofstream ofs(output_file);
  if (VERBOSE) cout << "  Copula support  of (" << fset << ") = " << computeFreq(rankn, fset) << endl;

  if (fpm) {
    ts = clock();
    runFPM(rankn, fset, freq_current, -1, n, size_limit, ofs);
    te = clock();
    if (!VERBOSE) {
      cout << "end" << endl << flush;
      sfst << "end" << endl << flush;
    }
    cout << "  # frequent combinations:    " << NUM_PATTERN << endl << flush;
    sfst << "  # frequent combinations:    " << NUM_PATTERN << endl;
    cout << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl << flush;
    sfst << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl;
    exit(1);
  }


  set<pair<double, double>, cmp> freq_pval_list;
  // freq_pval_list.insert(make_pair(0.0, 1e-20));
  SIGMA = 0.0;
  ADMISSIBLE = 1e20;
  ts = clock();
  double alpha_corrected;
  if (!bonferroni) {
    // =====
    // lamp
    // =====
    runSPM(rankn, fset, freq_current, -1, n, size_limit, ofs, N0, freq_pval_list, alpha);
    alpha_corrected = alpha / NUM_PATTERN;
  } else {
    // =====
    // Bonferroni
    // =====
    SIGMA = 0.0;
    runFPM(rankn, fset, freq_current, -1, n, size_limit, ofs);
    NUM_PATTERN += 1.0;
    alpha_corrected = alpha / NUM_PATTERN;
  }
  // Int num_testable = NUM_PATTERN;
  te = clock();
  if (!VERBOSE) {
    cout << "end" << endl << flush;
    sfst << "end" << endl << flush;
  }
  cout << "  # testable combinations:    " << NUM_PATTERN << endl << flush;
  sfst << "  # testable combinations:    " << NUM_PATTERN << endl;
  cout << "  Corrected alpha:            " << alpha_corrected << endl << flush;
  sfst << "  Corrected alpha:            " << alpha_corrected << endl;
  cout << "  Frequency threshold:        " << SIGMA << endl << flush;
  sfst << "  Frequency threshold:        " << SIGMA << endl;
  cout << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl << flush;
  sfst << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl;


  // ------------------------------------------------------------------------- //
  // ---------- Enumerate significant patterns with corrected alpha ---------- //
  // ------------------------------------------------------------------------- //
  cout << "> Find significant combinations with a threshold " << SIGMA << endl << flush;
  sfst << "> Find significant combinations with a threshold " << SIGMA << endl;
  NUM_PATTERN = 0.0;
  ts = clock();
  for (auto&& x : freq_current) x = 1.0;
  runSPM_sig(rankn, fset, freq_current, -1, n, size_limit, N0, cl, alpha_corrected, ofs);
  te = clock();
  cout << "  # significant combinations: " << NUM_PATTERN << endl << flush;
  sfst << "  # significant combinations: " << NUM_PATTERN << endl;
  cout << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl;
  sfst << "  Running time:               " << (float)(te - ts) / CLOCKS_PER_SEC << " [sec]" << endl;
}
