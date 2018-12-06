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

/*********************************************************************************************//**
 * 	\file MolecularDynamics.h
 * 	\brief Molecular dynamics integrators to find updated particle positions.
 *
 * 	I should write a detailed explanation at some point.
 *
 * 	\author Dillon Cislo
 * 	\date 11/16/2018
 * 	\copyright GNU Public License
 *
 ************************************************************************************************/

#ifndef _MOLECULAR_DYNAMICS_H_
#define _MOLECULAR_DYNAMICS_H_

#include <Eigen/Core>
#include <stdexcept> // std::runtime_error
#include <iostream>
#include <cmath>

namespace FIREpp {

///
/// Molecular dynamics integrators to find updated system vectors. Mainly for internal use.
///
template <typename Scalar>
class MolecularDynamics {

	private:

		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> 		Vector;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> 	Matrix;
		typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> 	Array;

	public:

		///
		/// Molecular dynamics via velocity verlet integration
		///
		/// \param f 		A function object such that `f(x, grad)` returns the
		/// 			objective function value at `x`, and overwrites `grad`
		///			with the gradient.
		/// \param fx 		IN: The objective function value at the current point.
		///			OUT: The function value at the new point.
		/// \param x 		IN: The current point.
		///			OUT: The new point.
		/// \param v 		IN: The current velocity.
		/// 			OUT: The new velocity.
		/// \param grad		IN: The gradient at the current point.
		/// 			OUT: The gradient at the new point.
		/// \param dt 		The current time step
		///
		/// \param alpha 	The current steering parameter
		///
		/// \param m 		The vector of particle masses
		///
		/// \param param	Parameters for the FIRE algorithm
		///
		template <typename Foo>
		static void VelocityVerlet( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Check input arguments
			if ( dt <= Scalar(0) )
				std::invalid_argument("'dt' must be positive");
			if ( alpha <= Scalar(0) )
				std::invalid_argument("'alpha' must be positive");

			// Choose appropriate integration algorithm
			switch( param.boundary_conditions ) {

				case FIRE_NO_BOUNDARY_CONDITIONS : 
					
					VelocityVerlet_NBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

				case FIRE_PERIODIC_BOUNDARY_CONDITIONS :
					
					VelocityVerlet_PBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

				case FIRE_HARD_BOUNDARY_CONDITIONS :
					
					VelocityVerlet_HBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

			}

		};

		//! Velocity verlet integration with no boundary conditions
		template <typename Foo>
		static void VelocityVerlet_NBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Perform the first set of position/velocity updates 
			Vector a = -( grad.array() / m.array() ).matrix();
			x +=  dt * v + dt * dt * a / Scalar(2.0);
			v +=  dt * a / Scalar(2.0);

			// Update gradient vector
			fx = f( x, grad );

			// Perform final velocity updates
			a = -( grad.array() / m.array() ).matrix();
			v += dt * a / Scalar(2.0);
			
		};

		//! Velocity verlet integration with periodic boundary conditions
		template <typename Foo>
		static void VelocityVerlet_PBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Extract bounding box
			Vector lbnd = param.lbnd;
			Vector ubnd = param.ubnd;
			Vector len = ubnd - lbnd;

			// Perform the first set of position/velocity updates 
			Vector a = -( grad.array() / m.array() ).matrix();
			x +=  dt * v + dt * dt * a / Scalar(2.0);
			v +=  dt * a / Scalar(2.0);

			// Reset positions within the periodic bounding box
			for( int i = 0; i < param.n_p; i++ ) {
				for( int j = 0; j < param.dim; j++ ) {


					x(i+j*param.n_p) -= len(j) * 
					std::floor( ( x(i+j*param.n_p)-lbnd(j) ) / len(j) );

				}

			}

			// Update gradient vector
			fx = f( x, grad );

			// Perform final velocity updates
			a = -( grad.array() / m.array() ).matrix();
			v += dt * a / Scalar(2.0);

		};

		//! Velocity verlet integration with hard wall boundary conditions
		template <typename Foo>
		static void VelocityVerlet_HBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Extract bounding box
			Vector lbnd = param.lbnd;
			Vector ubnd = param.ubnd;
			Vector len = ubnd - lbnd;

			// Perform the first set of position/velocity updates
			Vector a = -( grad.array() / m.array() ).matrix();
			x += dt * v + dt * dt * a / Scalar(2.0);
			v += dt * a / Scalar(2.0);

			// Reset positions and velocities within the hard wall bounding box
			for( int i = 0; i < param.n_p; i++ ) {
				for( int j = 0; j < param.dim; j++ ) {

					Scalar dx = Scalar(0.0);
					if ( x(i+j*param.n_p) > ubnd(j) ) {

						dx = x(i+j*param.n_p) - ubnd(j);
						dx -= len(j) *
						       	std::floor( (dx-lbnd(j)) / len(j) );
						x(i+j*param.n_p) = ubnd(j) - dx;

						v(i+j*param.n_p) *= Scalar(-1.0);

					} else if ( x(i+j*param.n_p) < lbnd(j) ) {

						dx = lbnd(j) - x(i+j*param.n_p);
						dx -= len(j) *
							std::floor( (dx-lbnd(j)) / len(j) );
						x(i+j*param.n_p) = lbnd(j) + dx;

						v(i+j*param.n_p) *= Scalar(-1.0);

					}

				}
			}

			// Update gradient vector
			fx = f(x, grad);

			// Perform final velocity updates
			a = -( grad.array() / m.array() ).matrix();
			v += dt * a / Scalar(2.0);

		};


};

} // namespace FIREpp

#endif // _MOLECULAR_DYNAMICS_H_
