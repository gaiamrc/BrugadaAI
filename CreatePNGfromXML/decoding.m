function leads = decoding(record,dir)

    leads = {}; % This will ultimately hold the extracted data.
    fid = fopen([dir record]); %Asssign file id and open file
    data = fscanf(fid,'%c'); %Import as character array
    
    % find boundaries of <parsedwaveforms> element
    front_hit_header = strfind(data,'<parsedwaveforms'); %Find starting landmark for 12-Lead Data
    grthns = strfind(data,'>'); %In 3 steps, find the end of this XML field
    grlocs = find(grthns > front_hit_header); % Step 2...
    front_hit = grthns(grlocs(1)); %Step 3...
    back_hit = strfind(data,'</parsedwaveforms>'); %Find ending landmark for 12-Lead Data
    
    % extract Base64 encoded data
    data = data(front_hit+1:back_hit-1); %Crop 12-Lead Data
    data(strfind(data,' ')) = []; %Remove structural spaces (maybe not necessary)
    data(strfind(data,sprintf('\n'))) = []; %Remove newlines
    data(strfind(data,sprintf('\r'))) = []; %Remove line feeds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Decode from Base64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Note: Use Octave's base64decode
    decoded = uint8(base64decode(data));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Extract each of the 12 leads
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    leadOffset = 0;
    for n = 1:12
    
        % Extract chunk header
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The header is the first 64 bits of each chunk.
        header = decoded(leadOffset+1:leadOffset+8);
    
        % The size of the ECG data following it is coded in the first 32 bits.
        datasize = typecast(header(1:4), 'uint32');
    
        % The second part of the header is a 16bit integer of unknown purpose.
        codeone = typecast(header(5:6), 'uint16'); % That integer converted from binary.
    
        % The last part of the header is a signed 16bit integer that we will use later (delta code #1).
        delta = typecast(header(7:8), 'int16');
    
        % Now we use datasize above to read the appropriate number of bytes
        % beyond the header. This is encoded ECG data.
        block = uint8(decoded(leadOffset+9:leadOffset+9+datasize-1));
        % assert(datasize == length(block));
    
        % Convert 8-bit bytes into 10-bit codes (stored in 16-bit ints)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % number of 10-bit codes
        codecount = floor((datasize*8)/10);
        codes = zeros(1, codecount, 'uint16');
    
        offset = 1;
        bitsRead = 0;
        buffer = uint32(0);
        done = false;
        for code = 1:codecount
            % adapted from libsierraecg
            while bitsRead <= 24
              if offset > datasize
                  done = true;
                  break;
              else
                  buffer = bitor(buffer, bitshift(uint32(block(offset)), 24 - bitsRead));
                  offset = offset + 1;
                  bitsRead = bitsRead + 8;
              end
            end
    
            if done
                break;
            else
                % 32 - codeSize = 22
                codes(code) = uint16(bitand(bitshift(buffer, -22), 65535));
                buffer = bitshift(buffer, 10);
                bitsRead = bitsRead - 10;
            end
        end
    
        % LZW Decompression
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Data is compressed with 10-bit LZW codes (last 2 codes are padding)
        [decomp, table] = lzw2norm(codes(1:length(codes)-2));
    
        %If the array length is not a multiple of 2, tack on a zero.
        if mod(length(decomp),2)~=0
            decomp = [decomp 0];
        end
    
        % Deinterleave into signed 16-bit integers
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The decompressed data is stored [HIWORDS...LOWORDS]
        half = length(decomp)/2;
        output = reshape([decomp(half+1:length(decomp));decomp(1:half)],1,[]);
        output = typecast(output, 'int16');
    
        % The 16bit ints are delta codes. We now use the delta decoding scheme
        % outlined by Watford to reconstitute the original signal.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        first = delta;
        prev = first;
        x = output(1);
        y = output(2);
        z = zeros(length(output),1);
        z(1) = x;
        z(2) = y;
        for m = 3:length(output)
            z(m) = (2*y)-x-prev;
            prev = output(m) - 64;
            x = y;
            y = z(m);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        leads = [leads z];
    
        % move to the next lead (8 byte header + datasize bytes of payload)
        leadOffset = leadOffset + 8 + datasize;
    end
    
    % Convert leads cell array to numeric matrix
    leads = cell2mat(leads);
    
    % Reconstruct Lead III, aVR, aVL, aVF
    leads(:,3) = leads(:,2) - leads(:,1) - leads(:,3);
    leads(:,4) = -leads(:,4) - (leads(:,1) + leads(:,2))/2;
    leads(:,5) = (leads(:,1) - leads(:,3))/2 - leads(:,5);
    leads(:,6) = (leads(:,2) + leads(:,3))/2 - leads(:,6);
end