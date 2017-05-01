#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

#define pi 3.1415926

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  // improve these following codes according to review suggestions. 
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  //MatrixXd Si = S.inverse();
  MatrixXd K = PHt * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  // no influence on float or double for these following parameters
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

  //pre-compute a set of terms to avoid repeated calculation
  //ignore unecessary c1
  //double c1 = px*px+py*py;
  double c2 = sqrt(px*px+py*py);
  double phi = atan2(x_[1], x_[0]);
  //avoid division By Zero
  if (fabs(c2) < 0.0001) {
    c2 = 0.0001;
  }
  double c3 = (x_[0] * x_[2] + x_[1] * x_[3]) / c2;
  Eigen::VectorXd z_pred = VectorXd(3);
  z_pred << c2, phi, c3;
	
  VectorXd y = z - z_pred;

  //angle normalization
  while (y(1)> pi) y(1)-=2.*pi;
  while (y(1)<-pi) y(1)+=2.*pi;
  
  // improve these following codes according to review suggestions.
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  //MatrixXd Si = S.inverse();
  MatrixXd K = PHt * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
