#pragma once

#ifndef __DISTRIBUTION_H_
#define __DISTRIBUTION_H_

#include "includes.h"

class distribution
{
public:
    int b;
    std::vector<float> quantiles;

    distribution();
    distribution(std::vector<float> quantiles_para)
    {
        this->quantiles = quantiles_para;
        this->b = quantiles_para.size();
    }
    distribution(torch::Tensor quantiles_para) // {b}
    {
        // int p = quantiles_para.size(0);
        int b = quantiles_para.numel(); // check this?
        float* ptr = (float*)quantiles_para.data_ptr();
        std::vector<float> quantiles;
        for(int i =0; i<b ; i++)
        {
            quantiles.push_back(*ptr);
            ptr++;
        }
        sort(quantiles.begin(),quantiles.end());
        this->b = b;
        this->quantiles = quantiles;
        this->b = quantiles.size();
    }
    
    distribution(event_type sampler,int b_para,int64_t n= 100000)
    {
        this->b = b_para;
        // Generate sorted samples
        // Apply H&L Method
        /*
        n - samples
        p - percentile 0...100 100 groups (percentile) <-> b groups (b quantile)
        METHOD 10 (“H&L-2”):  The  Pth  percentile value is found by taking that value with  #(np + 0.5).  If this is not an integer, take the interpolated value between    and  # .  As an example, if  
        S5 = (1, 2, 3, 4, 5) and  p = .25, then  #(np + 0.5) = #(1.75) and so  Q1 = 1.75.  
        */
        std::vector< std::pair<float,float> > samples;
        for (size_t i = 0; i < n; i++)
        {
            samples.push_back( {sampler(0), 0} );
        }
        std::sort(samples.begin(),samples.end());

        for (size_t i = 0; i < n; i++)
        {
            // std::cout<<samples[i].first <<',';
            samples[i].second = (i+0.5)/n;
        }
        //    0...100 <-> b groups kth b-quantile 

        for (size_t k = 1; k <= b; k++)
        {
            // ((j-1)-0.5)/n <-> (j-0.5)/n
            // ((j-1)-0.5)/n <= k*100/b
            // (k*100/b)
            int j = int( 1.0*k*n/b + 1.5 );
            quantiles.push_back( (samples[j-1].first - samples[j].first)/(samples[j-1].second - samples[j].second) * ( 1.0*k/b - samples[j].second ) + samples[j].first );

            // simpler Algo
            // if( (k*b)%n == 0)
            // {
            //     quantiles.push_back( (samples[k*b/n].first + samples[k*b/n+1].first)/2 );
            // }
            // else
            //     quantiles.push_back( samples[int(k*b/n)+1].first );
        }        
        this->b = quantiles.size();
    }
    
    distribution(std::vector<float> data,int b_para)
    {
        
        this->b = b_para;

        int n = data.size();
        std::vector< std::pair<float,float> > samples;
        for (size_t i = 0; i < n; i++)
        {
            samples.push_back( { data[i], 0} );
        }

        std::sort(samples.begin(),samples.end());

        for (size_t i = 0; i < n; i++)
        {
            // std::cout<<samples[i].first <<',';
            samples[i].second = (i+0.5)/n;
        }
        //    0...100 <-> b groups kth b-quantile 

        for (size_t k = 1; k <= b; k++)
        {
            // ((j-1)-0.5)/n <-> (j-0.5)/n
            // ((j-1)-0.5)/n <= k*100/b
            // (k*100/b)
            int j = int( 1.0*k*n/b + 1.5 );
            quantiles.push_back( (samples[j-1].first - samples[j].first)/(samples[j-1].second - samples[j].second) * ( 1.0*k/b - samples[j].second ) + samples[j].first );

            // simpler Algo
            // if( (k*b)%n == 0)
            // {
            //     quantiles.push_back( (samples[k*b/n].first + samples[k*b/n+1].first)/2 );
            // }
            // else
            //     quantiles.push_back( samples[int(k*b/n)+1].first );
        }        
        this->b = quantiles.size();
    }

    void print_quantiles()
    {
        std::cout<<'[';
        for(auto&x: quantiles)
        {
            std::cout<<x <<',';
        }
        std::cout<<']';
    }

    // A better continous sampler can be constructed
    event_type sampler()
    {
        int b_para = this->b;
        std::vector<float> quantiles_para = this->quantiles;
        return [b_para,quantiles_para](float t) -> float { float U = random; return quantiles_para[int(b_para*U)]; };
    }

    static float area_between_dist(distribution A,distribution B)
    {
        assert(A.quantiles.size()==B.quantiles.size());
        float res = 0;
        for (size_t i = 0; i < A.quantiles.size(); i++)
        {
            res += abs(A.quantiles[i] - B.quantiles[i]);
        }
        res /= A.b;
        return res;
    }
    torch::Tensor convert_to_tensor()
    {
        int len = quantiles.size();
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        return torch::from_blob(quantiles.data(), {1,len}, options);
        // return torch::tensor(quantiles.data());
    }
    float calc_epsilon()
    {
        float epsilon = INF;
        for(int i=1;i<quantiles.size();i++)
        {
            if( epsilon > quantiles[i]-quantiles[i-1])
                epsilon = quantiles[i] - quantiles[i-1];

        }
        return epsilon;
    }
    void distort(torch::Tensor distortion) // b
    {
        float epsilon = calc_epsilon();
        
        std::vector<float> distortion_list = convert_tensor<float>(distortion); // -1 to 1
        for(int i=0;i<quantiles.size();i++)
        {
            quantiles[i] += epsilon*distortion_list[i];
        }
    }
};

#endif