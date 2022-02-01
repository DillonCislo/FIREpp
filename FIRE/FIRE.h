/*********************************************************************************************** 
 * Copyright (C) 2018 Dillon Cislo
 *
 * This file is part of FIRE++.
 *
 * FIRE++ is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 ***********************************************************************************************/

/*******************************************************************************************//**
 * 	\file FIRE.h
 * 	\brief A FIRE solver for numerical optimization
 *
 * 	I should write a detailed explanation at some point.
 *
 * 	\author Dillon Cislo
 * 	\date 11/16/2018
 * 	\copyright GNU Public License
 *
 ***********************************************************************************************/


#ifndef _FIRE_H_
#define _FIRE_H_

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cfloat>
#include <cmath>

#include <Eigen/Core>

#include "Param.h"
#include "MolecularDynamics.h"

namespace FIREpp {


///
/// FIRE solver for unconstrained numerical optimization
///
template <typename Scalar>
class FIRESolver {

	private:

		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

		const FIREParam<Scalar> &m_param;   // Parameters to control the FIRE algorithm

		Vector 			m_fx;       // History of objective function values
		Vector 			m_xp; 	    // Old x
		Vector 			m_grad;     // New gradient
		Vector      m_gradp;    // Old gradient
		Vector 			m_v;        // New velocity
		Vector 			m_vp; 	    // Old velocity
		Vector 			m_mass;     // Particle masses
		bool 		  	m_userMass; // Check whether the user supplied masses

		inline void reset( int n ) {

			m_xp.resize(n);
			m_grad.resize(n);
			m_gradp.resize(n);

			m_v.resize(n);
			m_v = Vector::Zero(n);
			m_vp.resize(n);
			m_vp = Vector::Zero(n);

			if ( m_param.past > 0 )
				m_fx.resize(m_param.past);
		};

	public:

		///
		/// Constructor for FIRE solver
		///
		/// \param param An object of \ref FIREParam to store parameters
		/// 	for the algorithm
		///
		FIRESolver( const FIREParam<Scalar> &param ) : m_param( param ) {
			m_param.check_param();
			m_userMass = false;
		};

		///
		/// Constructor for FIRE solver with mass
		///
		/// \param param An object of \ref FIREParam to store parameters
		/// 	for the algorithm
		/// \param mass A vector of particle masses
		///
		FIRESolver( const FIREParam<Scalar> &param, Vector mass ) :
		       	m_param( param ), m_mass( mass ) {

			m_param.check_param();
			m_userMass = true;

		};

		///
		/// Minimizing a multivariate function using the FIRE algorithm
		/// Exceptions will be thrown if error occurs.
		///
		/// \param f	A function object such that `f(x, grad)` returns the
		///		objective function value at `x`, and overwrites `grad`
		/// 		with the gradient.
		/// \param x 	IN: An initial guress of the optimal point.
		///		OUT: The best point found
		/// \param fx 	OUT: The objective function value at `x`.
		///
		/// \return 	Number of iterations used
		///
		template <typename Foo>
		inline int minimize( Foo &f, Vector &x, Scalar &fx ) {

			const int n = x.size();
			const int fpast = m_param.past;
			reset(n);
			if ( !m_userMass ) {
				m_mass.resize(n);
				m_mass = Vector::Constant(n,1,Scalar(1.0));
			}

			// Evaluate function and compute gradient
			fx = f(x, m_grad);
			Scalar xnorm = x.norm();
			Scalar gnorm = m_grad.norm();
			if( fpast > 0 )
				m_fx[0] = fx;

      int k = 0;

      // Display iterative updates
      if (m_param.iter_display) {

        std::cout << "( 0)"
          << " ||dx|| = " << std::scientific << std::setw(10)
          << std::setprecision(5) << gnorm
          << " ||x|| = " << std::setprecision(5)
          << std::setw(10) << xnorm
          << " f(x) = " << std::setw(10) << std::setprecision(5)
          << fx << std::endl;

      }

      // Handle NaNs produced by the initial guess
      if ( (fx != fx) || (xnorm != xnorm) || (gnorm != gnorm) )
        throw std::invalid_argument("Initial guess generates NaNs");

      // Handle Infs produced by the initial guess
      if ( (std::isinf(fx)) || std::isinf(xnorm) || std::isinf(gnorm) )
        throw std::invalid_argument("Initial guess generates Infs");

			// Early exit if the initial x is already a minimizer
			if( gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * xnorm ) {

        if ( m_param.iter_display )
          std::cout << "EARLY EXIT CONDITION: Gradient Norm" << std::endl;

        return 1;

      }

			k++;
			int steps_since_freeze = 0;
			Scalar dt = m_param.dt_start;
			Scalar alpha = m_param.alpha_start;

			while( true ) {

				// Save the current x, v, and gradient
				m_xp.noalias() = x;
				m_gradp.noalias() = m_grad;
				m_vp.noalias() = m_v;

				// Molecular dynamics update to current system configuration
				MolecularDynamics<Scalar>::VelocityVerlet( f, fx,
						x, m_v, m_grad, dt, alpha, m_mass, m_param );

				// New x norm and gradient norm
				xnorm = x.norm();
				gnorm = m_grad.norm();

        // Handle NaNs produced by the current iterate
        if ( (fx != fx) || (xnorm != xnorm) || (gnorm != gnorm) )
          throw std::invalid_argument("Current iterate generates NaNs");

        // Handle Infs produced by the current interate
        if ( (std::isinf(fx)) || std::isinf(xnorm) || std::isinf(gnorm) )
          throw std::invalid_argument("Current iterate generates Infs");

 	      // Display iterative updates
				if (m_param.iter_display) {

          std::cout << "(" << std::setw(2) << k << ")"
            << " ||dx|| = " << std::scientific << std::setw(10)
       	    << std::setprecision(5) << gnorm
            << " ||x|| = " << std::setprecision(5)
					  << std::setw(10) << xnorm
            << " f(x) = " << std::setw(10) << std::setprecision(5)
            << fx << std::endl;

				}

				// Convergence test -- gradient
				if ( gnorm <= m_param.epsilon || gnorm <= m_param.epsilon_rel * xnorm ) {

          if ( m_param.iter_display )
            std::cout << "CONVERGENCE CRITERION: Gradient Norm" << std::endl;

          return k;

        }

				// Convergence test -- objective function value
				if ( fpast > 0 ) {

					if ( (k >= fpast) &&
					(std::abs((m_fx[k % fpast]-fx)/fx) < m_param.delta) ) {

            if ( m_param.iter_display )
              std::cout << "CONVERGENCE CRITERION: Objective Function Value" << std::endl;

						return k;

					}

					m_fx[ k % fpast ] = fx;

				}

				// Maximum number of iterations
				if(m_param.max_iterations != 0 && k >= m_param.max_iterations) {

          if ( m_param.iter_display )
            std::cout << "CONVERGENCE CRITERION: Maximum Iteration Count" << std::endl;

					return k;

				}

				Scalar P = m_v.dot( -m_grad );

				m_v = ( Scalar(1)-alpha ) * m_v +
					alpha * m_v.norm() * ( -m_grad ) / m_grad.norm();

				// If the current velocity is 'downhill'
				if ( P > Scalar(0) ) {

					if ( steps_since_freeze > m_param.nmin ) {

						alpha = alpha * m_param.falpha;
						dt = std::min( dt * m_param.finc,
								m_param.dt_max );

					}

					steps_since_freeze++;

				// If the current velocity is 'uphill'
				} else {

					steps_since_freeze = 0;

					dt = dt * m_param.fdec;
					alpha = m_param.alpha_start;

					m_v = Vector::Zero(n);

				}

				k++;

			}

			return k;

		};

};

} // namespace FIREpp

#endif // _FIRE_H_

