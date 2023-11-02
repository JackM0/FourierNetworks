"""
Script used to execute the discrete periodic Radon transform

Author: Marlon Bran Lorenzana
"""

"""
This class will contain functions for the forward and inverse discrete periodic Radon transform
"""
import torch
import math
import radon.farey as farey #local module

pi = torch.tensor(math.pi)
##############################################################################
###        Pytorch nn.Module functions to perform the drt and idrt         ###
############################################################################## 

"""
Forward projection of the discrete periodic Radon transform used for NN layer
"""
class drt(torch.nn.Module):

    def __init__(self):
        super(drt, self).__init__()
        self.drt_forback = dpradon()

    
    def forward(self, image, mValues=None, center=False, norm='ortho'):
        """
        Pass through to the dpradon class for evaluation
        """
        return self.drt_forback.frt(image, mValues=mValues, center=center, norm=norm)

    

"""
Back projection of the discrete periodic Radon transform used for NN layer
"""
class idrt(torch.nn.Module):

    def __init__(self):
        super(idrt, self).__init__()
        self.idrt_forback = dpradon()

    
    def forward(self, sino, mValues=None, center=False, norm='ortho'):
        """
        Pass through to the dpradon class for evaluation
        """
        return self.idrt_forback.ifrt(sino, mValues=mValues, center=center, norm=norm)

##############################################################################
###  Pytorch nn.Module functions to perform the drt and idrt more quickly  ###
##############################################################################

# To speed this up and fit within a single torch.nn.Module will instead accept
# as an input, an array of indices, so M x N -> each containing the co-ords at
# M lines. This way we can Image -> FFT2 -> Extract locations -> iFFT on each 
# set or FFT each set -> Place in locations -> iFFT2. 

"""
Forward projection of the discrete periodic Radon transform used for NN layer

Uses pre-calculated locations for given mValues

Mojette option allows for computing pre-calculated locations that are physically
realistic
"""
class fdrt(torch.nn.Module):

    def __init__(self, N, mValues=None, pad=True, center=False, ridgelet=False, norm='ortho', mojette=False):
        super(fdrt, self).__init__()
        # Get image shape
        self.N = N
        # Get number of projections
        if N % 2 == 0:
            self.mu = int(N + N / 2)
        else:
            self.mu = N + 1
        #create discrete aperiodic coords
        self.mojette = mojette
        if self.mojette:
            #create angle set with Farey vectors
            fareyVectors = farey.Farey()        
            fareyVectors.compactOn()
            fareyVectors.generateFiniteWithCoverage(N, L1Norm=False)
            self.angles = fareyVectors.vectors
            self.mValues = fareyVectors.finiteAngles
            if not mValues is None: #use only mValues provided for aperiodic coords
                angles = []
                for m in mValues:
                    index = self.mValues.index(m)
                    angles.append(self.angles[index])
                self.angles = angles
        # Populate mValues
        if mValues is None:
            self.mValues = range(0, self.mu)
        else:
            self.mValues = mValues
        # Get number of mValues
        self.lenM = len(self.mValues)
        # Populate lines
        if not self.mojette:
            self.lines = dpradon.slices.getSlicesCoordinates(N, mValues=self.mValues, ridgelet=ridgelet,
                                                                                    center=center)
        else:
            self.lines = dpradon.slices.getFareySlicesCoordinates(N, angles=self.angles, ridgelet=ridgelet,
                                                                                    center=center)
        # Get norm type
        self.norm = norm
        # Get centre type
        self.center = center
        # Get padding type
        self.pad = pad
        
    def forward(self, image):
        """
        Accept only an image and perform the drt for pre-specified mValues.

        Returns sinogram with zeros where no mValues have been specified.
        """
        # Check if image is given as a set of channels rather than complex
        if image.dim() > 3 and image.shape[1] < 2:
            image = image.permute(0, 2, 3, 1).contiguous()
            image = torch.cat([image, torch.zeros_like(image)], dim=3)
            image = torch.view_as_complex(image)

        # Get shape of image
        batch, _, _ = image.shape

        # Get dftSpace
        dftSpace = torch.fft.fft2(image, norm=self.norm)

        # Create sino array
        if self.pad:
            sino = torch.zeros(batch, self.mu, self.N, dtype=dftSpace.dtype,
                                                            device=image.device)
        
        # Produce discrete periodic sinogram
        if self.pad:
            sino[:, self.mValues, :] = torch.fft.ifft(dftSpace[:, self.lines[:, 0, :],
                                                        self.lines[:, 1, :]], norm=self.norm)
        else:
            sino = torch.fft.ifft(dftSpace[:, self.lines[:, 0, :], self.lines[:, 1, :]], norm=self.norm)

        # Return discrete periodic sinogram
        return sino

"""
Back projection of the discrete periodic Radon transform used for NN layer

Uses pre-calculated locations for given mValues
"""
class fidrt(torch.nn.Module):

    def __init__(self, N, mValues=None, pad=True, center=False, ridgelet=False, norm='ortho', mojette=False):
        super(fidrt, self).__init__()
        # Get image shape
        self.N = N
        # Get number of projections
        if N % 2 == 0:
            self.mu = int(N + N / 2)
            self.oversample = True
        else:
            self.mu = N + 1
            self.oversample = False
        #create discrete aperiodic coords
        self.mojette = mojette
        if self.mojette:
            #create angle set with Farey vectors
            fareyVectors = farey.Farey()        
            fareyVectors.compactOn()
            fareyVectors.generateFiniteWithCoverage(N, L1Norm=False)
            self.angles = fareyVectors.vectors
            self.mValues = fareyVectors.finiteAngles
            if not mValues is None: #use only mValues provided for aperiodic coords
                angles = []
                for m in mValues:
                    index = self.mValues.index(m)
                    angles.append(self.angles[index])
                self.angles = angles
        # Populate mValues
        if mValues is None:
            self.mValues = range(0, self.mu)
        else:
            self.mValues = mValues
        self.lenM = len(self.mValues)
        # Populate lines
        if not self.mojette:
            self.lines = dpradon.slices.getSlicesCoordinates(N, mValues=self.mValues, ridgelet=ridgelet,
                                                                                    center=center)
        else:
            self.lines = dpradon.slices.getFareySlicesCoordinates(N, angles=self.angles, ridgelet=ridgelet,
                                                                                    center=center)
        # Populate oversampleFilter
        self.oversampleFilter = dpradon.slices.oversamplingFilter(N, mValues=mValues, center=center)
        # Get oversample locations
        self.oversampleLocations = (self.oversampleFilter != 1).nonzero(as_tuple=False)
        # Get norm type
        self.norm = norm
        # Get centre type
        self.center = center
        # Get padding type
        self.pad = pad

    def forward(self, sino):
        """
        Accept only the discrete periodic sinogram for a pre-defined set of mValues.

        Returns image from sinogram.
        """
        # Check if sinogram is given as a set of channels rather than complex
        if sino.dim() > 3 and sino.shape[1] < 2:
            sino = sino.permute(0, 2, 3, 1).contiguous()
            sino = torch.cat([sino, torch.zeros_like(sino)], dim=3)
            sino = torch.view_as_complex(sino)

        # Get shape of sinogram
        batch, _, _ = sino.shape
        device = sino.device

        # Get the fft of the slices
        if self.pad:
            # Define slices array
            slices = torch.zeros_like(sino) # (batch, self.mu, self.N, dtype=torch.cfloat, device=device)
            slices[:, self.mValues, :] = torch.fft.fft(sino[:, self.mValues, :], norm=self.norm)
        else:
            slices = torch.fft.fft(sino, norm=self.norm)

        # Produce array to store dftSpace
        dftSpace = torch.zeros(batch, self.N, self.N, dtype=slices.dtype, device=device)

        # Place data into dftSpace
        if self.oversample: # If we can't update all dftSpace locations at once 
            for i, m in enumerate(self.mValues):
                # Get dft space values from slices
                if self.pad:
                    dftSpace[:, self.lines[i, 0, :], self.lines[i, 1, :]] += slices[:, m, :]
                else:
                    dftSpace[:, self.lines[i, 0, :], self.lines[i, 1, :]] += slices[:, i, :]

            # Divide by the oversampleFilter to account for the oversampling
            dftSpace /= self.oversampleFilter
        else: # (only if prime)
            # If no oversample required just place values in without taking average
            if self.pad:
                dftSpace[:, self.lines[:, 0, :], self.lines[:, 1, :]] = slices[:, self.mValues, :]
                dftSpace[:, 0, 0] = torch.mean(slices[:, self.mValues, 0])
            else:
                dftSpace[:, self.lines[:, 0, :], self.lines[:, 1, :]] = slices
                dftSpace[:, 0, 0] = torch.mean(slices[:, :, 0])

        # Return dftSpace 
        return torch.fft.ifft2(dftSpace, norm=self.norm)
    
    """
    Derive the cpu and cuda and to functions to ensure all tensors move to the allocated device
    """
    def cpu(self):
        self = super().cpu()
        self.oversampleFilter = self.oversampleFilter.cpu()
        self.oversampleLocations = self.oversampleLocations.cpu()
        return self
    
    def cuda(self, device=None):
        self = super().cuda(device)
        self.oversampleFilter = self.oversampleFilter.cuda(device)
        self.oversampleLocations = self.oversampleLocations.cuda(device)
        return self 

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.oversampleFilter = self.oversampleFilter.to(*args, **kwargs) 
        self.oversampleLocations = self.oversampleLocations.to(*args, **kwargs) 
        return self

"""
Forward discrete Fourier slice theorem used for NN layer

Assumes Fourier space (FFT of the image) is input

Uses pre-calculated locations for given mValues

Mojette option allows for computing pre-calculated locations that are physically
realistic
"""
class fst(torch.nn.Module):

    def __init__(self, N, mValues=None, pad=True, center=False, ridgelet=False, norm='ortho', mojette=False):
        super(fst, self).__init__()
        # Get image shape
        self.N = N
        # Get number of projections
        if N % 2 == 0:
            self.mu = int(N + N / 2)
        else:
            self.mu = N + 1
        #create discrete aperiodic coords
        self.mojette = mojette
        if self.mojette:
            #create angle set with Farey vectors
            fareyVectors = farey.Farey()        
            fareyVectors.compactOn()
            fareyVectors.generateFiniteWithCoverage(N, L1Norm=False)
            self.angles = fareyVectors.vectors
            self.mValues = fareyVectors.finiteAngles
            if not mValues is None: #use only mValues provided for aperiodic coords
                angles = []
                for m in mValues:
                    index = self.mValues.index(m)
                    angles.append(self.angles[index])
                self.angles = angles
        # Populate mValues
        if mValues is None:
            self.mValues = range(0, self.mu)
        else:
            self.mValues = mValues
        # Get number of mValues
        self.lenM = len(self.mValues)
        # Populate lines
        if not self.mojette:
            self.lines = dpradon.slices.getSlicesCoordinates(N, mValues=self.mValues, ridgelet=ridgelet,
                                                                                    center=center)
        else:
            self.lines = dpradon.slices.getFareySlicesCoordinates(N, angles=self.angles, ridgelet=ridgelet,
                                                                                    center=center)
        # Get norm type
        self.norm = norm
        # Get centre type
        self.center = center
        # Get padding type
        self.pad = pad
        
    def forward(self, dftSpace):
        """
        Accept only an image and perform the drt for pre-specified mValues.

        Returns sinogram with zeros where no mValues have been specified.
        """
        # Get shape of image
        batch, _, _ = dftSpace.shape

        # Create sino array
        if self.pad:
            sino = torch.zeros(batch, self.mu, self.N, dtype=dftSpace.dtype,
                                                            device=dftSpace.device)
        
        # Produce discrete periodic sinogram
        if self.pad:
            sino[:, self.mValues, :] = dftSpace[:, self.lines[:, 0, :], self.lines[:, 1, :]]
        else:
            sino = dftSpace[:, self.lines[:, 0, :], self.lines[:, 1, :]]

        if self.center:
            sino = torch.fft.fftshift(sino, axis=-1)

        # Return discrete periodic sinogram
        return sino

##############################################################################
###   Helper functions to perform the discrete periodic Radon transforms   ###
##############################################################################                                                             

class dpradon:
    """
    Class used to evaluate discrete periodic radon transform functions.

    Assumptions - Variables are [batch, rows, cols] and dtype=torch.cfloat.
                  Or [batch, numCh, rows, cols] and numCh = 2 for imag/real.
                - N x N image.

    Can convert to/from [batch, numCh, rows, cols] by using torch.view_as_real/complex,
    this requires torch.view_as_complex([batch, rows, cols, numCh]) to work. 
    """
    @classmethod
    def frt(cls, image, mValues=None, pad=True, ridgelet=False, center=False, norm='ortho'):
        """
        Function performs discrete Radon transform of an image. If no mValues
        given then DRT of full image will be computed.

        Always returns sinogram that achieves full coverage (zeros where no mValue).

        Inputs  - image: N x N Image in Cartesian grid
                - mValues: List of discrete gradients (m)
        Outputs - sino: M x N discrete periodic sinogram
        """
        if image.dim() > 3 and image.dtype is not torch.cfloat: # Check if image is given as a set of channels
            image = image.permute(0, 2, 3, 1).contiguous()
            image = torch.cat([image, torch.zeros_like(image)], dim=3)
            image = torch.view_as_complex(image)

        batch, N, _ = image.shape

        # Determine number of projections required for full coverage
        if N % 2 == 0:
            mu = int(N + N / 2)
        else:
            mu = N + 1

        # Produce mValues if none have been given
        if mValues is None:
            mValues = range(0, mu)

        # FFT Image
        dftSpace = torch.fft.fft2(image, norm=norm)

        if pad:
            # Create slices array
            slices = torch.zeros(batch, mu, N, dtype=dftSpace.dtype)

            # Get fft slices at mValue lines
            slices[:, mValues, :] = cls.slices.getSlices(dftSpace, mValues=mValues, ridgelet=ridgelet, center=center)
        else:
            # Get fft slices at mValue lines
            slices = cls.slices.getSlices(dftSpace, mValues=mValues, ridgelet=ridgelet, center=center)

        # Produce discrete periodic sinogram
        sino = torch.fft.ifft(slices, norm=norm)

        # Return sinogram populated at specified mValues
        return sino

    @classmethod
    def ifrt(cls, sino, mValues=None, oversampleFilter=None, pad=True, ridgelet=False, center=False, norm='ortho'):
        """
        Function performs inverse discrete Radon transform of an image
        if no mValues given then iDRT will be computed for all possible
        values.

        Expects size of discrete periodic sinogram that achieves full coverage.

        Inputs  - sino: M x N discrete periodic sinogram
                - mValues: List of discrete gradients (m)
        Outputs - image: N x N Image in Cartesian grid
        """
        if sino.dim() > 3 and sino.dtype is not torch.cfloat: # Check if sinogram is given as a set of channels
            sino = sino.permute(0, 2, 3, 1).contiguous()
            sino = torch.cat([sino, torch.zeros_like(sino)], dim=3)
            sino = torch.view_as_complex(sino)

        _, mLen, N = sino.shape

        # Produce mValues if none have been given
        if mValues is None:
            mValues = range(0, mLen)

        # Get dft space values from slices, setSlices expects data to be in dftSpace
        if pad:
            dftSpace = cls.slices.setSlices(torch.fft.fft(sino[:, mValues, :], norm=norm), mValues=mValues, ridgelet=ridgelet, center=center)
        else:
            dftSpace = cls.slices.setSlices(torch.fft.fft(sino, norm=norm), mValues=mValues, ridgelet=ridgelet, center=center)

        # Divide by the oversampleFilter to account for the oversampling
        if oversampleFilter is None:
            oversampleFilter = cls.slices.oversamplingFilter(N, mValues=mValues, center=center)
        dftSpace /= oversampleFilter

        # Return dftSpace 
        return torch.fft.ifft2(dftSpace, norm=norm)

    
    class slices:
        """
        Class to perform the slice manipulation required to perform DRT and iDRT.
        """
        @classmethod
        def getSlices(cls, data, mValues=None, ridgelet=False, center=False):
            """
            Get slices at given mValues of an N x N discrete arary using the
            discrete Fourier slice theorem. This can be applied to the DFT or NTT.

            Returns zeros where mValue not specified
            """
            batch, N, _ = data.shape

            # Produce mValues if none are given
            if mValues is None:
                if N % 2 == 0:
                    mu = int(N + N / 2)
                else:
                    mu = N + 1
                mValues = range(0, mu) 

            # Create array to store the projections 
            slices = torch.zeros(batch, len(mValues), N, dtype=data.dtype)

            # Get each of the projection locations for given mValues
            for i, m in enumerate(mValues):
                slices[:, i, :] = cls.getSlice(m, data, ridgelet=ridgelet, center=center)

            # Return the slices 
            return slices

        @classmethod
        def setSlices(cls, slices, mValues=None, ridgelet=False, center=False):
            """
            Set slices at given mValues for an M x N discrete periodic sinogram
            """
            batch, _, N = slices.shape

            # Produce mValues if none are given
            if mValues is None:
                if N % 2 == 0:
                    mu = int(N + N / 2)
                else:
                    mu = N + 1
                mValues = range(0, mu)            

            # Create an N x N array to set the slices to
            data = torch.zeros(batch, N, N, dtype=slices.dtype)

            # Iterate through given m values and set the slice
            for i, m in enumerate(mValues):
                cls.setSlice(m, data, slices[:, i, :], ridgelet=ridgelet, center=center)

            # Return data from given slices
            return data

        @classmethod
        def getSlicesCoordinates(cls, N, mValues=None, ridgelet=True, center=False):
            """
            Gets the slice co-ordinates for given m values for an N x N basis
            """
            # Produce mValues if none are given
            if mValues is None:
                if N % 2 == 0:
                    mu = int(N + N / 2)
                else:
                    mu = N + 1
                mValues = range(0, mu) 

            # Create an array to store co-ordinate locations
            lines = torch.zeros(len(mValues), 2, N, dtype=torch.long)

            # Get the co-ordinate location for each of the m values
            for i, m in enumerate(mValues):
                if ridgelet:
                    lines[i, :, :] = cls.getDiscreteLine(m, N, center=center)
                else:
                    lines[i, :, :] = cls.getSliceCoordinates(m, N, center=center)

            # Return the array of lines
            return lines

        @classmethod
        def getFareySlicesCoordinates(cls, N, angles, ridgelet=True, center=False):
            """
            Gets the slice co-ordinates for given m values for an N x N basis
            """
            # Create an array to store co-ordinate locations
            lines = torch.zeros(len(angles), 2, N, dtype=torch.long)

            # Get the co-ordinate location for each of the m values
            for i, angle in enumerate(angles):
                lines[i, :, :] = cls.getFareySliceCoordinates(angle, N, center=center)

            # Return the array of lines
            return lines

        @classmethod
        def getSlice(cls, m, data, ridgelet=False, center=False):
            """
            Get the slice (at m) of an N x N discrete array using the discrete 
            Fourier slice theorem. This can be applied to the DFT or the NTT arrays.
            """
            _, N, _ = data.shape

            # Get slice co-ordinates for given m value
            if ridgelet:
                line = cls.getDiscreteLine(m, N, center=center)
            else:
                line = cls.getSliceCoordinates(m, N, center=center)

            # Create slice
            slice = data[:, line[0, :], line[1, :]]

            # Return the slice
            return slice
            
        @classmethod
        def setSlice(cls, m, data, slice, ridgelet=False, center=False):
            """
            Place slice data (for values of m) of a discrete periodic sinogram using
            the discrete Fourier slice theorem. This can be applied to the DFT or NTT arrays.
            """
            _, N, _ = data.shape

            # Get slice co-ordinates for given m value
            if ridgelet:
                line = cls.getDiscreteLine(m, N, center=center)
            else:
                line = cls.getSliceCoordinates(m, N, center=center)

            # Place slice into data
            data[:, line[0, :], line[1, :]] += slice

        @classmethod
        def oversamplingFilter(cls, N, mValues=None, center=False):
            """
            Populates an N x N matrix with oversampling filter

            Returns NaN in zero locations to easily apply
            """

            # Produce mValues if none are given
            if N % 2 == 0:
                mu = int(N + N / 2)
            else:
                mu = N + 1

            # Populate mValues
            if mValues is None:
                mValues = range(0, mu)

            # Create N x N array to store filter
            oversampleFilter = torch.zeros(N, N)

            # Get all slice locations for all possible gradients
            for m in mValues:
                # Get the location for the line at gradient m
                line = cls.getSliceCoordinates(m, N, center=center)
                # Add to oversample filter
                oversampleFilter[line[0, :], line[1, :]] += 1

            # Fill zeros with ones to help with division
            oversampleFilter[oversampleFilter == 0] = 1

            # Return the oversample filter
            return oversampleFilter

        def getSliceCoordinates(m, N, center=False):
            """
            Gets the slice co-ordinates u, v arrays (in pixel co-ordinates at finite angle m)
            of an N x N discrete array using the discrete Fourier slice theorem. 
            """
            line = torch.zeros(2, N, dtype=torch.long)

            if N % 2 == 0:
                p = 2
            else:
                p = N

            # Account for centering
            if center:
                offset = int(N / 2.0)
            else:
                offset = 0

            if m < N and m >= 0: # Whether m or s value
                for k in range(0, N):
                    # Produce x
                    x = (k + offset) % N
                    # Generate index for data
                    index = (((k * m) % N) + offset) % N
                    # Store locations
                    line[0, k] = x
                    line[1, k] = index
            else: # Perpendicular slice or dyadic slices
                s = m - N # Consistent notation
                for k in range(0, N):
                    # Produce x
                    x = (k + offset) % N
                    # Generate index for image
                    index = (((k * p * s) % N) + offset) % N
                    # Extract slice component
                    line[1, k] = x
                    line[0, k] = index

            if center:
                line = torch.fft.fftshift(line, dim=-1)

            # Return the list of locations for the particular line 
            return line

        def getFareySliceCoordinates(angle, N, center=False):
            """
            Gets the slice co-ordinates u, v arrays (in pixel co-ordinates at finite angle m)
            of an N x N discrete array using the discrete Fourier slice theorem. 
            """
            line = torch.zeros(2, N, dtype=torch.long)

            # Account for centering
            if center:
                offset = int(N / 2.0)
            else:
                offset = 0
    
            p, q = farey.get_pq(angle)
            if p < 0:
                p += N
            if q < 0:
                q += N

            # indices = []
            for k in range(0,N):
                y = (N-(k*p+offset)%N)%N
                x = (N-(k*q+offset)%N)%N
                # indices.append((x,y)) 
                line[1, k] = x
                line[0, k] = y

            # Return the list of locations for the particular line 
            return line

        @classmethod
        def getDiscreteAngles(cls, N, mValues=None, radians=True):
            """
            Gets the discrete angles for given mValues
            """
            # Produce mValues if none are given
            if N % 2 == 0:
                mu = int(N + N / 2)
            else:
                mu = N + 1

            # Populate mValues
            if mValues is None:
                mValues = range(0, mu)

            angles = torch.zeros(len(mValues))

            for i, m in enumerate(mValues):
                angles[i] = cls.getDiscreteAngle(N, m, radians)

            return angles

        @classmethod
        def getDiscreteAngle(cls, N, m, radians=True):
            """
            Function converts discrete gradient m into equivalent angle.

            Inputs      - N:    Size of the image (N x N)
                        - m:    Angle required
            Outputs     - ang:  Angles
            """
            y, x = cls.getClosestPoint(N, m)

            # Get the radial angle
            angle = torch.atan(torch.tensor(y), torch.tensor(x))

            if m >= N or m < 0:
                angle += pi / 2

            if not radians:
                angle = torch.rad2deg(angle)

            return angle

        @classmethod
        def getDiscreteLine(cls, m, N, center=False):

            # Get the first point 
            y, x = cls.getClosestPoint(N, m)

            """
            Gets the slice co-ordinates u, v arrays (in pixel co-ordinates at finite angle m)
            of an N x N discrete array using the discrete Fourier slice theorem. 
            """
            line = torch.zeros([2, N], dtype=torch.long)

            # Account for centering
            if center:
                offset = int(N / 2.0)
            else:
                offset = 0

            for i in range(0, N):
                line[0, i] = (x * i + offset) % N
                line[1, i] = (y * i + offset) % N

            if center:
                line = torch.fft.fftshift(line, dim=-1)

            # Return the list of locations for the particular line 
            return line

        def getClosestPoint(N, m):
            """
            Function returns the closest point.

            Inputs      - N:    Size of the image (N x N)
                        - m:    Angle required
            Outputs     - x, y: Closest Point
            """
            # Account for centering
            offset = int(N / 2.0)

            # Get dimension-space
            if N % 2 == 0:
                p = 2
            else:
                p = N

            # Search for the lowest value
            inds = torch.tensor(range(1, offset + 1))
            if m < N and m >= 0: # Whether m or s value
                indsy = torch.remainder((inds * m + offset), N) - offset
            else: # Perpendicular slice or dyadic slices
                s = m - N # Consistent notation
                indsy = torch.clone(inds)
                inds = torch.remainder(torch.remainder(inds * p * s, N) + offset, N) - offset
            
            ind = torch.argmin(torch.sqrt(inds**2 + indsy**2))
            x = inds[ind]
            y = indsy[ind]
            return y, x