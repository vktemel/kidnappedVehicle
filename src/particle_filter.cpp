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
  // Calculate the change in yaw and distance by multiplying with delta_t
  double yaw_change = yaw_rate*delta_t;
  double distance = velocity*delta_t;

  // create normal distributions for adding noise based on standard deviations
  std::normal_distribution<double> dist_x{0, std_pos[0]};
  std::normal_distribution<double> dist_y{0, std_pos[1]};
  std::normal_distribution<double> dist_theta{0, std_pos[2]};

  // prediction will be done for each particle
  for(auto& particle : particles)
  {
    // if yaw rate is not equal to zero, consider change in yaw rate in updating of particles
    if(yaw_rate != 0.0F)
    {
      particle.x = particle.x + (velocity/yaw_rate) * (sin(particle.theta + yaw_change) - sin(particle.theta)) + dist_x(engine);
      particle.y = particle.y + (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_change)) + dist_y(engine);
      particle.theta = particle.theta + yaw_change + dist_theta(engine);
    }
    // otherwise (if yaw rate is zero), then update particles with distance travelled on the current heading
    else
    {
      particle.x = particle.x + distance*cos(particle.theta) + dist_x(engine);
      particle.y = particle.y + distance*sin(particle.theta) + dist_y(engine);
      // although there's yaw rate is 0, theta is updated with the noise component
      particle.theta = particle.theta + dist_theta(engine); 
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
  // Unused.
  

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // Weights will be updated for each particle
  for(auto& particle : particles)
  {
    // First step is to transform the observations for the given particle
    // Create a vector for transformed observations
    vector<LandmarkObs> transformed_observations;

    // Transformation will be applied to each observation
    for(auto& obs : observations)
    {
      // Create a temporary Landmark observation for transformation
      LandmarkObs temp; 
      
      // Transform x and y according to the heading of the particle
      temp.x = particle.x + (obs.x*cos(particle.theta) - obs.y*sin(particle.theta));
      temp.y = particle.y + (obs.y*cos(particle.theta) + obs.x*sin(particle.theta));
      
      // Add to transformed observations vector
      transformed_observations.push_back(temp); 
    }

    // To make function more efficient, only landmarks within the sensor range will be considered.
    // Vehicle sensor range is assumed to be equal for all 360 degree around the vehicle
    vector<LandmarkObs> relevant_landmarks;
    for(auto& landmark : map_landmarks.landmark_list)
    {
      // calculate distance to landmark using the helper function
      double dist_to_landmark = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      // if the landmark is within the sensor range of the particle, add to relevant landmarks
      if(dist_to_landmark <= sensor_range)
      {
        LandmarkObs temp;
        temp.x = landmark.x_f;
        temp.y = landmark.y_f; 
        temp.id = landmark.id_i;
        relevant_landmarks.push_back(temp);
      }
    }

    // Update weights for the particle
    // First, create a temporary variable for new weight
    double t_weight = 1.0;
    
    // Create temporary vectors for associations and sensed x and y
    std::vector<int> t_associations;
    std::vector<double> t_sense_x, t_sense_y;
    
    // Updating weights for particle will be done for each observation
    for(auto& obs : transformed_observations)
    {
      // First step is to find the closest landmark for the given observation
      int id_closest;
      // Iniitalize minimum distance to a very high value
      double min_dist = 1e8;
      double x_val, y_val;

      // Find the closest landmark for this observation by checking distance to each landmark
      for(auto& landmark : relevant_landmarks)
      {
        // Calculate the distance using helper function
        double temp_dist = dist(landmark.x, landmark.y, obs.x, obs.y);

        // If the newly calculated distance is smaller than the previously found minimum distance
        // Store the properties of that landmark and update the minimum distance
        // Otherwise, ignore the landmark
        if(temp_dist < min_dist)
        {
          min_dist = temp_dist;
          id_closest = landmark.id;
          x_val = landmark.x;
          y_val = landmark.y;
        }  
      }

      // Update the id of the observation, and add the associations and sensed x, y to the vectors
      obs.id = id_closest; 
      t_associations.push_back(id_closest);
      t_sense_x.push_back(x_val);
      t_sense_y.push_back(y_val);

      // Calculate the weight using the multivariate Gaussian distribution, based on the properties
      // of the the observation and the sigma values of the landmark
      double var_x = std_landmark[0]*std_landmark[0];
      double var_y = std_landmark[1]*std_landmark[1];
      double covar_xy = std_landmark[0]*std_landmark[1];

      double diff_x = obs.x - x_val;
      double diff_y = obs.y - y_val;
      double norm = (1/(2*M_PI*covar_xy));

      double obs_weight = norm*exp(-1*((diff_x*diff_x)/(2*var_x) + (diff_y*diff_y)/(2*var_y)));

      // weight of the particle will be product of all calculated weights
      t_weight *= obs_weight;

    }
    // set the associations and the sensed values to the particle
    SetAssociations(particle, t_associations, t_sense_x, t_sense_y);

    // update particle weight
    particle.weight = t_weight;
  }

}

void ParticleFilter::resample() {
  
  // Resampling will be done proportional to the weight of each particle
  // First create a vector that stores weights of each particle
  vector<double> particle_weights;
  for(auto& particle : particles)
  {
    particle_weights.push_back(particle.weight);
  }

  // Create a discrete distribution based on the particle weights
  std::discrete_distribution<int> d{particle_weights.begin(),particle_weights.end()};  

  // Create a vector for new particles that are sampled accordingly
  std::vector<Particle> new_particles; 
  for(int i=0; i<num_particles; i++)
  {
    // Add particle based on the result of the discrete distribution called by the random engine
    new_particles.push_back(particles[d(engine)]);
  }

  // set particles to the newly selected particles
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