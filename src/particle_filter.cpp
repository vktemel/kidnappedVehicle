/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * This function receives the initial x, y, theta positions and their standard
   * deviations, and then creates a list of particles. The particles are created
   * by adding a random Gaussian noise to the given initialization values for x,
   * y and theta. The number of particles created is defined by the num_particles
   * parameter. 
   */
  
  // Set the number of particles
  num_particles = 20;

  // Use the default random engine
  std::default_random_engine engine;
  // create normal distributions for x, y, and theta based on inputs of this function
  // these will be used to add random Gaussian noise for each particle
  std::normal_distribution<double> dist_x{x, std[0]};
  std::normal_distribution<double> dist_y{y, std[1]};
  std::normal_distribution<double> dist_theta{theta, std[2]};

  // add num_particles particles in initialization
  for(int i=0; i<num_particles; i++)
  {
    // create a temporary particle
    Particle t_particle;

    // assign properties of the particle
    t_particle.id = i;
    t_particle.x = dist_x(engine);
    t_particle.y = dist_y(engine);
    t_particle.theta = dist_theta(engine);
    t_particle.weight = double(1.0);

    // add the temporary particle to the overall vector of particles
    particles.push_back(t_particle);
    // add the weight to the vector of weights
    weights.push_back(t_particle.weight);
  }

  // set initialization flag to true
  is_initialized = true; 

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // if yaw rate is positive
  // xf = x0 + v/thetadot (sin (theta+thetadot*dt) - sin(theta))
  // yf = y0 + v/thetadot (cos(theta) - cos(theta+thetadot*dt))
  // thetaf = theta + thetadot*dt
  double yaw_change = yaw_rate*delta_t;
  double distance = velocity*delta_t;
  std::default_random_engine engine;
  std::normal_distribution<double> dist_x{0, std_pos[0]};
  std::normal_distribution<double> dist_y{0, std_pos[1]};
  std::normal_distribution<double> dist_theta{0, std_pos[2]};

  for(int i=0; i<num_particles; i++)
  {
    if(yaw_rate != 0.0F)
    {
      particles[i].x = (particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + yaw_change) - sin(particles[i].theta))) + dist_x(engine);
      particles[i].y = (particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_change))) + dist_y(engine);
      particles[i].theta = (particles[i].theta + yaw_change) +	 dist_theta(engine);
    }
    else
    {
      particles[i].x = particles[i].x + distance*cos(particles[i].theta);
      particles[i].y = particles[i].y + distance*sin(particles[i].theta);
      // no change in theta
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // for each particle
    // calculate weights of each observation
    // get product of all weights
    // update weights of the particle
  for(auto& particle : particles)
  {
    vector<LandmarkObs> transformed_observations;
    // transform each observation to map coordinates
    for(auto& obs : observations)
    {
      LandmarkObs temp; 
      temp.x = particle.x + (obs.x*cos(particle.theta) - obs.y*sin(particle.theta));
      temp.y = particle.y + (obs.y*cos(particle.theta) + obs.x*sin(particle.theta));
      transformed_observations.push_back(temp); 
    }

    // filter out relevant landmarks for the particle based on sensor_range
    vector<LandmarkObs> relevant_landmarks;
    for(auto& landmark : map_landmarks.landmark_list)
    {
      double dist_to_landmark = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      if(dist_to_landmark <= sensor_range)
      {
        LandmarkObs temp;
        temp.x = landmark.x_f;
        temp.y = landmark.y_f; 
        temp.id = landmark.id_i;
        relevant_landmarks.push_back(temp);
      }
    }

    // associate observations to landmarks
    //dataAssociation(relevant_landmarks, transformed_observations);
    vector<int> associations; 
    vector<double> sense_x_vals;
    vector<double> sense_y_vals;

    double new_weight = 1.0;
    
    std::vector<int> t_associations;
    std::vector<double> t_sense_x, t_sense_y;
    
    for(auto& obs : transformed_observations)
    {
      int id_closest;
      double min_dist = 1e8;
      double x_val;
      double y_val;

      for(auto& landmark : relevant_landmarks)
      {
        double temp_dist = dist(landmark.x, landmark.y, obs.x, obs.y);

        if(temp_dist < min_dist)
        {
          min_dist = temp_dist;
          id_closest = landmark.id;
          x_val = landmark.x;
          y_val = landmark.y;
        }  
      }

      obs.id = id_closest; 
      t_associations.push_back(id_closest);
      t_sense_x.push_back(x_val);
      t_sense_y.push_back(y_val);


      double var_x = std_landmark[0]*std_landmark[0];
      double var_y = std_landmark[1]*std_landmark[1];
      double covar_xy = std_landmark[0]*std_landmark[1];

      double diff_x = obs.x - x_val;
      double diff_y = obs.y - y_val;

      double temp_weight = (1/(2*M_PI*covar_xy))*exp(-1*((diff_x*diff_x)/(2*var_x) + (diff_y*diff_y)/(2*var_y)));

      new_weight *= temp_weight;

    }
    SetAssociations(particle, t_associations, t_sense_x, t_sense_y);
    particle.weight = new_weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> particle_weights;
  for(auto& particle : particles)
  {
    particle_weights.push_back(particle.weight);
  }

  std::default_random_engine engine; 
  std::discrete_distribution<int> d{particle_weights.begin(),particle_weights.end()};  

  std::vector<Particle> new_particles; 

  for(int i=0; i<num_particles; i++)
  {
    int new_particle_number = d(engine);
    new_particles.push_back(particles[new_particle_number]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}