/*!
 * 	\file Param.h
 * 	\brief Parameters to control the FIRE algorithm
 *
 * 	\author Dillon Cislo
 * 	\date 11/16/2018
 *
 */

#ifndef _PARAM_H_
#define _PARAM_H_

#include <Eigen/Core>
#include <stdexcept> // std::invalid_argument

namespace FIREpp {


///
/// 	\defgroup Enumerations
///
///	Enumeration types for declaring boundary conditions
///

///
/// 	\ingroup Enumerations
/// 	
///	The enumeration of the types of boundary conditions
///
enum BOUNDARY_CONDITIONS {

	//! No boundary conditions
	FIRE_NO_BOUNDARY_CONDITIONS = 1,

	//! Periodic boundary conditions
	FIRE_PERIODIC_BOUNDARY_CONDITIONS = 2,

	//! Hard wall boundary conditions
	FIRE_HARD_BOUNDARY_CONDITIONS = 3

};

///
/// Parameters to control the FIRE algorithm
///
template <typename Scalar = double>
class FIREParam {

	private:

		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> 	Vector;

	public:

		///
		/// The latency time to accelerate dynamics after freezing.
		/// Upon 'uphill motion', the system is frozen ( v <- 0 ).
		/// Requiring the system to wait a small number of steps before
		/// accelerating again is important for stability. Should be
		/// larget than one to maintain at least a few smooth steps
		/// after freezing.
		///
		int nmin;

		///
		/// The factor by which the timestep increases for each 'downhill' step
		/// Should be larger than, but near to one to avoid overly fact acceleration.
		///
		Scalar finc;

		///
		/// The factor by which the time step decreases upon an 'uphill' step.
		/// Should be smaller than 1, but much larger than zero to avoid
		/// overly heavy slow downs.
		///
		Scalar fdec;

		///
		/// The starting value for the steering coefficient that tends to send the
		/// system along a 'steeper' path.  Should be larger than, but near to zero
		/// to avoid overly heavy damping.
		///
		Scalar alpha_start;

		///
		/// The starting value for the time step
		///
		Scalar dt_start;

		///
		/// The factor by which the steering coefficient is decreased for each
		/// 'downhill' step.  Should be smaller than, but near to one so that
		/// mixing is efficient some time after restart from freezing.
		///
		Scalar falpha;

		///
		/// The maximum allowable timestep.  Highly system dependent.
		/// If molecular dynamics simulations of the system are available,
		/// the maximum time step should be approximately ten times the 
		/// MD timestep.
		///
		Scalar dt_max;

		///
		/// The type of boundary conditions obeyed by the system.
		///
		int boundary_conditions;

		///
		/// The maximum number of iterations for which the algorithm
		/// is allowed to run.
		///
		int max_iterations;

		///
		/// The gradient convergence criteria. The algorithim is stoppped
		/// when the norm of the gradient ||dx|| < epsilon.
		///
		Scalar epsilon;

		///
		/// Distance for delta-based convergence test.
		/// This parameter determines the distance \f$d\f$ to compute the
		/// rate of decrease of the objective function,
		/// \f$(f_{k-d}(x)-f_k(x))/f_k(x)\f$, where \f$k\f$ is the current iteration
		/// step. If the value of this parameter is zero, the delta-based convergence
		/// test will not be performed.
		///
		int past;

		///
		/// Delta for delta-based convergence test.
		/// The algorithm stops when the following condition is met:
		/// \f$(f_{k-d}(x)-f_k(x))/f_k(x)<\delta\f$, where \f$f_k(x)\f$ is
		/// the furrent function value, \f$f_{k-d}(x)\f$ is the function value
		/// \f$d\fS iterations ago (specified by the \ref past parameter).
		///
		Scalar delta;

		///
		/// The number of "particles" in the system
		///
		int n_p;

		///
		/// The dimensionality of the simulated system
		///
		int dim;

		///
		/// The minimum bound of the bounding box
		///
		Vector lbnd;

		///
		/// The maximum bound of the bounding box
		///
		Vector ubnd;

		///
		/// Iterative display option
		///
		bool iter_display;
		

	public:

		///
		/// Constructor for FIRE parameters
		/// Default values for parameters will be set when the object is created
		///
		FIREParam(int _n_p, int _dim) : n_p( _n_p), dim( _dim) {

			nmin = 5;
			finc = Scalar(1.1);
			fdec = Scalar(0.5);
			alpha_start = Scalar(0.1);
			dt_start = Scalar(0.01);
			falpha = Scalar(0.99);
			dt_max = Scalar(0.1);
			boundary_conditions = FIRE_NO_BOUNDARY_CONDITIONS;
			max_iterations = 0;
			epsilon = Scalar(1e-5);
			past = 0;
			delta = Scalar(0);
			iter_display = false;
			
			lbnd.resize(dim);
			lbnd = Vector::Constant(dim, Scalar(0));

			ubnd.resize(dim);
			ubnd = Vector::Constant(dim, Scalar(10));

		};

		///
		/// Checking the validity of the FIRE parameters
		/// An `std::invalid_argument` exception will be thrown if some parameter
		/// is invalid
		///
		inline void check_param() const {

			if ( nmin <= 0 )
				throw std::invalid_argument("'nmin' must be positive");
			if ( finc <= 1 )
				throw std::invalid_argument("'finc' must be greater than one");
			if ( (fdec <= 0) || (fdec >= 1) )
				throw std::invalid_argument("'fdec' must lie on (0,1)");
			if ( alpha_start <= 0 )
				throw std::invalid_argument("'alpha_start' must be positive");
			if ( (falpha <= 0) || (falpha >=1) )
				throw std::invalid_argument("'falpha' must lie on (0,1)");
			if ( dt_max <= 0 )
				throw std::invalid_argument("'dt_max' must be positive");
			if ( boundary_conditions < FIRE_NO_BOUNDARY_CONDITIONS ||
			boundary_conditions > FIRE_HARD_BOUNDARY_CONDITIONS )
				throw std::invalid_argument("Unsupported boundary condition");
			if ( max_iterations < 0 )
				throw std::invalid_argument("'max_iterations' must be >= 0");
			if ( epsilon <= 0 )
				throw std::invalid_argument("'epsilon' must be positive");
			if ( past < 0 )
				throw std::invalid_argument("'past' must be non-negative");
			if ( delta < 0 )
				throw std::invalid_argument("'delta' must be non-negative");
			if ( n_p <= 0 )
				throw std::invalid_argument("'n_p' must be positive");
			if ( dim <= 0 )
				throw std::invalid_argument("'dim' must be positive");
			if ( dt_start <= 0 )
				throw std::invalid_argument("'dt_start' must be positive");
			if ( lbnd.size() != dim )
				throw std::invalid_argument("'lbnd' improperly sized");
			if ( ubnd.size() != dim )
				throw std::invalid_argument("'ubnd' improperly sized");

			for( int i = 0; i < dim; i++ ) {
				if ( ubnd(i) <= lbnd(i) ) {
					throw std::invalid_argument("Invalid bounds");
				}
			}

		};

};

} // namespace FIREpp

#endif // _PARAM_H_
